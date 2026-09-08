#include "assoc/edge_store.hpp"

#include <utility>
#include <vector>

#include "assoc/csr.hpp"

namespace assoc {

// ---------------------------------------------------------------------------
// freeze
//
// The only point where the mutable and immutable halves meet. Scans the delta,
// drops edges that have decayed below the floor, builds a fresh CSR snapshot,
// and publishes it with a release store.
//
// The scan runs concurrently with writers. A slot may be read mid-bump, giving
// either the pre- or post-reinforcement value; both are valid, and the missing
// reinforcement lands in the next freeze. This is the staleness window, and it
// is acceptable precisely because Hebbian weights are statistical rather than
// transactional.
//
// Dead edges are excluded from the snapshot but left in the delta. Compaction
// would need to repair probe chains, which is deferred. Leaving them is also
// harmless: a later reinforce() finds the slot, decays it to ~0, and adds eta,
// so a resurrected edge starts fresh rather than inheriting a stale weight.
// ---------------------------------------------------------------------------

void EdgeStore::freeze(Tick tick) {
  std::vector<std::pair<EdgeKey, EdgeVal>> live;
  live.reserve(live_.load(std::memory_order_relaxed));

  std::size_t dropped = 0;

  for (std::size_t i = 0; i < capacity_; ++i) {
    const EdgeKey k = slots_[i].key.load(std::memory_order_acquire);
    if (k == kEmptyKey) continue;

    const EdgeVal v = slots_[i].val.load(std::memory_order_acquire);
    if (decay_.is_dead(decay_.decayed(v, tick))) {
      ++dropped;
      continue;
    }

    // Store the packed value as-is, not the decayed scalar. The snapshot
    // applies decay on read too, so pre-decaying here would double-apply it
    // and would also freeze the weight at this tick.
    live.emplace_back(k, v);
  }

  auto snap = std::make_shared<const CsrSnapshot>(
      CsrSnapshot::build(std::move(live), decay_.params()));

  std::atomic_store_explicit(&snapshot_, std::move(snap), std::memory_order_release);
  dropped_.fetch_add(dropped, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Query delegation
//
// Every query reads the published snapshot and nothing else, so none of them
// touch the delta and none can contend with a writer. Before the first
// freeze() the snapshot is null and queries return empty rather than faulting.
// ---------------------------------------------------------------------------

std::size_t EdgeStore::activate(const NodeId* seeds, const float* seed_w, std::size_t n_seeds,
                                std::uint32_t hops, float cutoff, Tick tick, NodeId* out_nodes,
                                float* out_scores, std::size_t cap) const {
  const auto snap = snapshot();
  if (!snap) return 0;
  return snap->activate(seeds, seed_w, n_seeds, hops, cutoff, tick, out_nodes, out_scores, cap);
}

void EdgeStore::weights_for_pairs(const NodeId* a, const NodeId* b, std::size_t n, Tick tick,
                                  float* out) const {
  const auto snap = snapshot();
  if (!snap) {
    for (std::size_t i = 0; i < n; ++i) out[i] = 0.0f;
    return;
  }
  snap->weights_for_pairs(a, b, n, tick, out);
}

float EdgeStore::edge_mass_along(const NodeId* ordered, std::size_t n, Tick tick) const {
  const auto snap = snapshot();
  return snap ? snap->edge_mass_along(ordered, n, tick) : 0.0f;
}

std::size_t EdgeStore::neighbors(NodeId node, float cutoff, Tick tick, NodeId* out, float* out_w,
                                 std::size_t cap) const {
  const auto snap = snapshot();
  return snap ? snap->neighbors(node, cutoff, tick, out, out_w, cap) : 0;
}

std::size_t EdgeStore::walk_directional(NodeId start, const float* dir, std::uint32_t hops,
                                        float alpha, Tick tick, NodeId* out_path,
                                        std::size_t cap) const {
  const auto snap = snapshot();
  if (!snap || emb_ == nullptr) return 0;
  // The walk should not step through edges that are already dead, so the
  // floor doubles as the traversal cutoff.
  return snap->walk_directional(start, dir, hops, alpha, decay_.params().floor, tick, emb_,
                                emb_nodes_, emb_dim_, out_path, cap);
}

}  // namespace assoc
