#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "assoc/decay.hpp"
#include "assoc/types.hpp"

namespace assoc {

// ---------------------------------------------------------------------------
// CsrSnapshot
//
// Immutable adjacency, published by freeze() and read by every query. Rows are
// contiguous and sorted by neighbour id: contiguity is what makes spreading
// activation prefetch-friendly, and sortedness turns a single-edge lookup into
// a binary search rather than a row scan.
//
// Edges in the delta are symmetric and stored once. Here each edge appears
// twice, once in each endpoint's row, so the snapshot holds 2E entries. That
// is the cost of making adjacency a contiguous span.
//
// Nothing mutates after build(), so concurrent readers need no synchronisation
// and freeze() publishes by swapping a shared_ptr rather than editing in place.
// ---------------------------------------------------------------------------

struct ActivationTransition {
  NodeId source, target;
  std::uint32_t hop;
  float contribution;
};

struct ActivationProfile {
  std::uint64_t row_visits = 0, unique_rows = 0, edge_visits = 0;
  std::uint64_t decay_evaluations = 0, merged_entries = 0, frontier_entries = 0;
  std::size_t max_frontier = 0;
  bool trace = false;
  std::vector<ActivationTransition> transitions;
};

class CsrSnapshot {
 public:
  // Takes live (key, value) pairs from a delta scan. Dead edges are expected
  // to have been dropped already by the caller, which owns the floor test.
  // Canonical keys sort by (lower endpoint, upper endpoint). For any row u,
  // reverse entries (a,u), a<u arrive in increasing a before forward entries
  // (u,b), b>u arrive in increasing b. Thus ONE sort of E canonical records
  // produces BOTH sorted directions without a temporary 2E directed array.
  // Input contract: canonical, unique, valid non-self edges from EdgeStore.
  static CsrSnapshot build(std::vector<std::pair<EdgeKey, EdgeVal>> live, Params p) {
    CsrSnapshot s(p);
    NodeId max_node = 0;
    for (const auto& [k,v] : live) max_node = std::max(max_node, key_hi(k));
    const std::size_t n = live.empty() ? 0 : static_cast<std::size_t>(max_node)+1;
    // For very sparse ID ranges, a cursor per node costs more than 2E entries.
    // Preserve the lower-workspace reference path in that regime.
    if (!live.empty() && n/4 > live.size()) return build_reference(std::move(live),p);
    std::sort(live.begin(), live.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    s.row_offsets_.assign(n+1,0);
    for (const auto& [k,v] : live) {
      ++s.row_offsets_[static_cast<std::size_t>(key_lo(k))+1];
      ++s.row_offsets_[static_cast<std::size_t>(key_hi(k))+1];
    }
    for (std::size_t i=1;i<=n;++i) s.row_offsets_[i]+=s.row_offsets_[i-1];
    auto next=s.row_offsets_;
    s.neighbors_.resize(live.size()*2);
    s.vals_.resize(live.size()*2);
    for (const auto& [k,v] : live) {
      const NodeId a=key_lo(k), b=key_hi(k);
      const auto ai=next[a]++, bi=next[b]++;
      s.neighbors_[ai]=b; s.vals_[ai]=v;
      s.neighbors_[bi]=a; s.vals_[bi]=v;
    }
    s.n_nodes_=n; s.n_edges_=live.size();
    return s;
  }

  // Reference retained for sparse-ID fallback, exact tests and benchmarks.
  static CsrSnapshot build_reference(std::vector<std::pair<EdgeKey, EdgeVal>> live, Params p) {
    CsrSnapshot s(p);

    NodeId max_node = 0;
    for (const auto& [k, v] : live) {
      max_node = std::max({max_node, key_lo(k), key_hi(k)});
    }
    const std::size_t n_nodes = live.empty() ? 0 : static_cast<std::size_t>(max_node) + 1;

    // Both directions, then one sort by (src, dst). Cheaper and simpler than
    // per-row sorting, and the result is exactly CSR fill order.
    std::vector<Entry> flat;
    flat.reserve(live.size() * 2);
    for (const auto& [k, v] : live) {
      const NodeId a = key_lo(k);
      const NodeId b = key_hi(k);
      flat.push_back({a, b, v});
      flat.push_back({b, a, v});
    }
    std::sort(flat.begin(), flat.end(), [](const Entry& x, const Entry& y) {
      return x.src != y.src ? x.src < y.src : x.dst < y.dst;
    });

    s.row_offsets_.assign(n_nodes + 1, 0);
    s.neighbors_.resize(flat.size());
    s.vals_.resize(flat.size());

    for (const auto& e : flat) ++s.row_offsets_[static_cast<std::size_t>(e.src) + 1];
    for (std::size_t i = 1; i <= n_nodes; ++i) s.row_offsets_[i] += s.row_offsets_[i - 1];

    for (std::size_t i = 0; i < flat.size(); ++i) {
      s.neighbors_[i] = flat[i].dst;
      s.vals_[i] = flat[i].val;
    }

    s.n_nodes_ = n_nodes;
    s.n_edges_ = live.size();
    return s;
  }

  // -------------------------------------------------------------------------
  // Row access
  // -------------------------------------------------------------------------

  [[nodiscard]] bool has(NodeId n) const noexcept {
    return static_cast<std::size_t>(n) < n_nodes_;
  }

  [[nodiscard]] std::span<const NodeId> row_nodes(NodeId n) const noexcept {
    if (!has(n)) return {};
    const std::size_t b = row_offsets_[n], e = row_offsets_[n + 1];
    return std::span<const NodeId>(neighbors_).subspan(b, e - b);
  }

  [[nodiscard]] std::span<const EdgeVal> row_vals(NodeId n) const noexcept {
    if (!has(n)) return {};
    const std::size_t b = row_offsets_[n], e = row_offsets_[n + 1];
    return std::span<const EdgeVal>(vals_).subspan(b, e - b);
  }

  [[nodiscard]] std::size_t degree(NodeId n) const noexcept {
    return has(n) ? row_offsets_[n + 1] - row_offsets_[n] : 0;
  }

  // -------------------------------------------------------------------------
  // Lookup queries. Thin enough to inline; all share find_weight.
  // -------------------------------------------------------------------------

  // Decayed weight of one edge, or 0 if it does not exist.
  [[nodiscard]] float find_weight(NodeId a, NodeId b, Tick tick) const noexcept {
    if (a == b || !has(a)) return 0.0f;
    const auto ns = row_nodes(a);
    if (ns.empty()) return 0.0f;
    const auto it = std::lower_bound(ns.begin(), ns.end(), b);
    if (it == ns.end() || *it != b) return 0.0f;
    return decay_.decayed(row_vals(a)[static_cast<std::size_t>(it - ns.begin())], tick);
  }

  // Engine half of structural-hole detection. Python supplies pairs that are
  // semantically close; a near-zero weight here means they were never
  // co-activated, which is a direction the conversation has not gone.
  void weights_for_pairs(const NodeId* a, const NodeId* b, std::size_t n, Tick tick,
                         float* out) const noexcept {
    for (std::size_t i = 0; i < n; ++i) out[i] = find_weight(a[i], b[i], tick);
  }

  // Sum of weights over consecutive pairs of an ordered sequence. Geometry
  // proposes an axis, this confirms whether usage actually travels it:
  // embedding spaces are full of accidental collinearity.
  [[nodiscard]] float edge_mass_along(const NodeId* ordered, std::size_t n,
                                      Tick tick) const noexcept {
    float mass = 0.0f;
    for (std::size_t i = 1; i < n; ++i) mass += find_weight(ordered[i - 1], ordered[i], tick);
    return mass;
  }

  // Raw decayed adjacency row above a cutoff. No geometry here: Python
  // projects these into a local tangent plane and bins them by angle, because
  // ambient angles in high dimensions are meaningless.
  std::size_t neighbors(NodeId node, float cutoff, Tick tick, NodeId* out, float* out_w,
                        std::size_t cap) const noexcept {
    const auto ns = row_nodes(node);
    const auto vs = row_vals(node);
    std::size_t k = 0;
    for (std::size_t i = 0; i < ns.size() && k < cap; ++i) {
      const float w = decay_.decayed(vs[i], tick);
      if (w < cutoff) continue;
      out[k] = ns[i];
      out_w[k] = w;
      ++k;
    }
    return k;
  }

  // -------------------------------------------------------------------------
  // Walking queries. Defined in csr.cpp; both need scratch buffers.
  // -------------------------------------------------------------------------

  // Spreading activation. The retrieval primitive: one sparse matvec per hop
  // with a cutoff applied between hops to keep the frontier sparse.
  std::size_t activate(const NodeId* seeds, const float* seed_w, std::size_t n_seeds,
                       std::uint32_t hops, float cutoff, Tick tick, NodeId* out_nodes,
                       float* out_scores, std::size_t cap,
                       ActivationProfile* profile = nullptr, bool optimized = true) const;

  // Curved traversal. Hops between real memories, steering by an accumulated
  // direction, rather than extrapolating a straight line off the manifold.
  // Embeddings are borrowed from the caller and row-major, dim per node.
  std::size_t walk_directional(NodeId start, const float* dir, std::uint32_t hops, float alpha,
                               float cutoff, Tick tick, const float* emb, std::size_t emb_nodes,
                               std::size_t dim, NodeId* out_path, std::size_t cap) const;

  // -------------------------------------------------------------------------

  [[nodiscard]] std::size_t node_count() const noexcept { return n_nodes_; }
  [[nodiscard]] std::size_t edge_count() const noexcept { return n_edges_; }
  [[nodiscard]] std::size_t entry_count() const noexcept { return neighbors_.size(); }
  [[nodiscard]] const DecayTable& decay() const noexcept { return decay_; }

  [[nodiscard]] std::size_t bytes() const noexcept {
    return row_offsets_.size() * sizeof(std::size_t) + neighbors_.size() * sizeof(NodeId) +
           vals_.size() * sizeof(EdgeVal);
  }

 private:
  struct Entry {
    NodeId src;
    NodeId dst;
    EdgeVal val;
  };

  explicit CsrSnapshot(Params p) : params_(p), decay_(p) {}

  Params params_{};
  DecayTable decay_{params_};
  std::vector<std::size_t> row_offsets_{};
  std::vector<NodeId> neighbors_{};
  std::vector<EdgeVal> vals_{};
  std::size_t n_nodes_ = 0;
  std::size_t n_edges_ = 0;
};

}  // namespace assoc
