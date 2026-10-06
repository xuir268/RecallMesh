#pragma once

#include <atomic>
#include <string>
#include <cstddef>
#include <memory>
#include <vector>

#include "assoc/decay.hpp"
#include "assoc/types.hpp"

namespace assoc {

class CsrSnapshot;  // csr.hpp; only needed complete inside edge_store.cpp

// ---------------------------------------------------------------------------
// Slot
//
// 16 bytes: four fit in a 64-byte cache line, eight in a 128-byte line.
// Key and value are separate atomics rather
// than one 128-bit word: they are never updated together, so a double-width
// CAS would buy nothing and is not lock-free everywhere.
// ---------------------------------------------------------------------------

struct Slot {
  std::atomic<EdgeKey> key;
  std::atomic<EdgeVal> val;
};

static_assert(sizeof(Slot) == 16, "slot must stay at 16 bytes");
static_assert(std::atomic<EdgeKey>::is_always_lock_free);

// ---------------------------------------------------------------------------
// EdgeStore
//
// Mutable hot delta plus a published immutable snapshot. reinforce() is the
// delta-update call; queries read a retained immutable snapshot. Publication
// uses atomic shared-pointer operations, which need not themselves be lock-free.
// ---------------------------------------------------------------------------

class EdgeStore {
 public:
  // Inserts are refused past this load factor. Resize is deliberately absent:
  // a growing lock-free table needs migration machinery that is not worth
  // building before the retrieval question is answered.
  static constexpr double kMaxLoad = 0.70;

  // Linear probing gives up after this many steps and counts a rejection,
  // rather than scanning the whole table under pathological clustering.
  static constexpr std::size_t kMaxProbe = 64;

  EdgeStore(std::size_t capacity_pow2, Params p)
      : decay_(p), capacity_(capacity_pow2), mask_(capacity_pow2 - 1), slots_(capacity_pow2) {
    if (capacity_pow2 < 2 || (capacity_pow2 & (capacity_pow2 - 1)))
      throw std::invalid_argument("capacity must be a power of two >= 2");
    for (auto& s : slots_) {
      s.key.store(kEmptyKey, std::memory_order_relaxed);
      s.val.store(0, std::memory_order_relaxed);
    }
  }

  EdgeStore(const EdgeStore&) = delete;
  EdgeStore& operator=(const EdgeStore&) = delete;

  // -------------------------------------------------------------------------
  // Write path
  //
  // Folds one co-activation batch. Probing is bounded; CAS retries are not,
  // so this update path is not wait-free. Counters accumulate locally and are published once per batch,
  // so the stats cache line is not itself a point of contention.
  // -------------------------------------------------------------------------
  void reinforce(const NodeId* src, const NodeId* dst, std::size_t n, Tick tick) {
    const float eta = decay_.params().eta;
    std::uint64_t ins = 0, upd = 0, retries = 0, probes = 0, rejected = 0;

    for (std::size_t i = 0; i < n; ++i) {
      const NodeId a = src[i];
      const NodeId b = dst[i];
      if (a == kInvalidNode || b == kInvalidNode || a == b) continue;  // no self-loops

      const EdgeKey key = pack_key(a, b);
      std::size_t idx = slot_of(key);
      bool placed = false;

      for (std::size_t probe = 0; probe < kMaxProbe; ) {
        const EdgeKey k = slots_[idx].key.load(std::memory_order_acquire);

        if (k == key) {
          bump(idx, tick, eta, retries);
          ++upd;
          probes += probe;
          placed = true;
          break;
        }

        if (k == kEmptyKey) {
          // Counters publish once per batch, so the load check must add this
          // batch's own inserts; without that a single large batch never sees
          // itself fill the table. Under concurrent writers other threads'
          // in-flight inserts are still invisible, so this is a soft bound —
          // kMaxProbe is the hard one.
          if (live_.load(std::memory_order_relaxed) + ins >= max_live()) break;

          EdgeKey expected = kEmptyKey;
          if (slots_[idx].key.compare_exchange_strong(expected, key,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_acquire)) {
            // Won the slot. val is still 0, but another thread may already be
            // reinforcing it, so seed through the same CAS loop rather than a
            // plain store: decay(0, tick) + eta == eta, and nothing is lost.
            bump(idx, tick, eta, retries);
            ++ins;
            probes += probe;
            placed = true;
            break;
          }
          // Lost the race. Re-examine this same slot without advancing: the
          // winner may have claimed our key, in which case the next iteration
          // takes the update branch.
          ++retries;
          continue;
        }

        idx = (idx + 1) & mask_;  // occupied by a different edge
        ++probe;
      }

      if (!placed) ++rejected;
    }

    if (ins) live_.fetch_add(ins, std::memory_order_relaxed);
    inserts_.fetch_add(ins, std::memory_order_relaxed);
    updates_.fetch_add(upd, std::memory_order_relaxed);
    cas_retries_.fetch_add(retries, std::memory_order_relaxed);
    probe_sum_.fetch_add(probes, std::memory_order_relaxed);
    rejected_.fetch_add(rejected, std::memory_order_relaxed);
  }

  // -------------------------------------------------------------------------
  // Snapshot lifecycle
  // -------------------------------------------------------------------------

  // Scan the delta, drop dead edges, build a fresh CSR snapshot, publish it.
  // Defined in edge_store.cpp, where CsrSnapshot is complete.
  void freeze(Tick tick);
  // Startup/checkpoint only: caller must exclude writers. Both files are immutable.
  void save_checkpoint(const std::string& path) const;
  void restore_checkpoint(const std::string& path);

  [[nodiscard]] std::shared_ptr<const CsrSnapshot> snapshot() const {
    return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
  }

  // Borrowed, read-only. The engine never owns, copies, or frees this.
  void attach_embeddings(const float* data, std::size_t n_nodes, std::size_t dim) {
    emb_ = data;
    emb_nodes_ = n_nodes;
    emb_dim_ = dim;
  }

  [[nodiscard]] std::size_t embedding_dim() const noexcept { return emb_dim_; }

  // -------------------------------------------------------------------------
  // Queries. Each delegates to the current snapshot; see csr.hpp.
  // -------------------------------------------------------------------------

  std::size_t activate(const NodeId* seeds, const float* seed_w, std::size_t n_seeds,
                       std::uint32_t hops, float cutoff, Tick tick,
                       NodeId* out_nodes, float* out_scores, std::size_t cap) const;

  void weights_for_pairs(const NodeId* a, const NodeId* b, std::size_t n, Tick tick,
                         float* out) const;

  float edge_mass_along(const NodeId* ordered, std::size_t n, Tick tick) const;

  std::size_t neighbors(NodeId node, float cutoff, Tick tick, NodeId* out, float* out_w,
                        std::size_t cap) const;

  std::size_t walk_directional(NodeId start, const float* dir, std::uint32_t hops,
                               float alpha, Tick tick, NodeId* out_path,
                               std::size_t cap) const;

  // -------------------------------------------------------------------------

  [[nodiscard]] Stats stats() const {
    Stats s;
    s.capacity = capacity_;
    s.live_edges = live_.load(std::memory_order_relaxed);
    s.inserts = inserts_.load(std::memory_order_relaxed);
    s.updates = updates_.load(std::memory_order_relaxed);
    s.cas_retries = cas_retries_.load(std::memory_order_relaxed);
    s.probe_length_sum = probe_sum_.load(std::memory_order_relaxed);
    s.rejected_full = rejected_.load(std::memory_order_relaxed);
    s.dropped_edges = dropped_.load(std::memory_order_relaxed);
    return s;
  }

  [[nodiscard]] const DecayTable& decay() const noexcept { return decay_; }

  // Decayed weight of one edge in the DELTA. Test and freeze use this; queries
  // read the snapshot instead.
  [[nodiscard]] float peek(NodeId a, NodeId b, Tick tick) const {
    if (a == b) return 0.0f;
    const EdgeKey key = pack_key(a, b);
    std::size_t idx = slot_of(key);
    for (std::size_t probe = 0; probe < kMaxProbe; ++probe, idx = (idx + 1) & mask_) {
      const EdgeKey k = slots_[idx].key.load(std::memory_order_acquire);
      if (k == key) return decay_.decayed(slots_[idx].val.load(std::memory_order_relaxed), tick);
      if (k == kEmptyKey) return 0.0f;
    }
    return 0.0f;
  }

 private:
  // splitmix64 finalizer. The packed key is two dense id ranges concatenated,
  // so the low bits alone cluster badly under a mask.
  [[nodiscard]] std::size_t slot_of(EdgeKey k) const noexcept {
    std::uint64_t x = k;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<std::size_t>(x) & mask_;
  }

  [[nodiscard]] std::size_t max_live() const noexcept {
    return static_cast<std::size_t>(static_cast<double>(capacity_) * kMaxLoad);
  }

  // The shared update: decay to now, add eta, swap. Correct under any
  // interleaving because the new value is a pure function of the observed one,
  // so a lost CAS just recomputes.
  void bump(std::size_t idx, Tick tick, float eta, std::uint64_t& retries) {
    EdgeVal v = slots_[idx].val.load(std::memory_order_relaxed);
    for (;;) {
      const EdgeVal nv = decay_.reinforce(v, tick, eta);
      if (slots_[idx].val.compare_exchange_weak(v, nv, std::memory_order_release,
                                                std::memory_order_relaxed)) {
        return;
      }
      ++retries;  // v now holds the current value; recompute from it
    }
  }

  DecayTable decay_;
  std::size_t capacity_;
  std::size_t mask_;
  std::vector<Slot> slots_;

  std::shared_ptr<const CsrSnapshot> snapshot_{};

  const float* emb_ = nullptr;
  std::size_t emb_nodes_ = 0;
  std::size_t emb_dim_ = 0;

  std::atomic<std::size_t> live_{0};
  std::atomic<std::uint64_t> inserts_{0};
  std::atomic<std::uint64_t> updates_{0};
  std::atomic<std::uint64_t> cas_retries_{0};
  std::atomic<std::uint64_t> probe_sum_{0};
  std::atomic<std::uint64_t> rejected_{0};
  // Edges excluded from a snapshot for falling below the floor. They stay in
  // the delta (no compaction yet), so live_ still counts their slots.
  std::atomic<std::uint64_t> dropped_{0};
};

}  // namespace assoc
