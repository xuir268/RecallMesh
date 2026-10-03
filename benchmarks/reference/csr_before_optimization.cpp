#include "assoc/csr.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace assoc {
namespace {

// Scratch reused across calls on the same thread. activate() is on the hot
// path and would otherwise allocate a node-sized accumulator per query.
struct Scratch {
  std::vector<float> accum;     // dense, indexed by NodeId
  std::vector<std::uint8_t> in_touched;
  std::vector<NodeId> touched;  // sparse reset list
  std::vector<NodeId> frontier;
  std::vector<float> front_w;
  std::vector<NodeId> next;
  std::vector<float> next_w;

  void prepare(std::size_t n_nodes) {
    if (accum.size() < n_nodes) {
      accum.assign(n_nodes, 0.0f);
      in_touched.assign(n_nodes, 0);
    }
    for (NodeId n : touched) {
      accum[n] = 0.0f;
      in_touched[n] = 0;
    }
    touched.clear();
    frontier.clear();
    front_w.clear();
  }

  void add(NodeId n, float w) {
    if (!in_touched[n]) {
      in_touched[n] = 1;
      touched.push_back(n);
    }
    accum[n] += w;
  }
};

Scratch& scratch() {
  thread_local Scratch s;
  return s;
}

[[nodiscard]] const float* emb_row(const float* emb, std::size_t dim, NodeId n) noexcept {
  return emb + static_cast<std::size_t>(n) * dim;
}

[[nodiscard]] float dot(const float* a, const float* b, std::size_t dim) noexcept {
  float s = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) s += a[i] * b[i];
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// activate
//
// Spreading activation: one sparse matvec per hop against the snapshot, with a
// cutoff between hops to keep the frontier sparse.
//
// Contributions are normalised by the source row's total mass, which makes
// each hop a random-walk transition rather than a raw weight product. Without
// it a hub node would dominate every result purely by degree, and activation
// would grow without bound across hops since weights saturate at eta/lambda
// rather than at 1.
//
// Scores accumulate across all hops, so a node reachable by several distinct
// paths outranks one reachable by a single path. That is the intended
// behaviour: convergent co-activation is stronger evidence than a single
// chain. Mass disperses with distance, so nearer nodes still dominate in
// general without needing an explicit per-hop damping term.
// ---------------------------------------------------------------------------

std::size_t CsrSnapshot::activate(const NodeId* seeds, const float* seed_w, std::size_t n_seeds,
                                  std::uint32_t hops, float cutoff, Tick tick, NodeId* out_nodes,
                                  float* out_scores, std::size_t cap) const {
  if (n_nodes_ == 0 || cap == 0) return 0;

  Scratch& s = scratch();
  s.prepare(n_nodes_);

  for (std::size_t i = 0; i < n_seeds; ++i) {
    const NodeId n = seeds[i];
    if (!has(n)) continue;
    const float w = seed_w ? seed_w[i] : 1.0f;
    if (w <= 0.0f) continue;
    s.add(n, w);
    s.frontier.push_back(n);
    s.front_w.push_back(w);
  }

  for (std::uint32_t h = 0; h < hops && !s.frontier.empty(); ++h) {
    s.next.clear();
    s.next_w.clear();

    for (std::size_t i = 0; i < s.frontier.size(); ++i) {
      const NodeId u = s.frontier[i];
      const float au = s.front_w[i];
      const auto ns = row_nodes(u);
      const auto vs = row_vals(u);
      if (ns.empty()) continue;

      // Row mass must be computed from decayed weights, not stored weights:
      // a row whose edges have aged unevenly would otherwise normalise wrong.
      float mass = 0.0f;
      for (std::size_t j = 0; j < vs.size(); ++j) mass += decay_.decayed(vs[j], tick);
      if (mass <= 0.0f) continue;

      for (std::size_t j = 0; j < ns.size(); ++j) {
        const float w = decay_.decayed(vs[j], tick);
        if (w <= 0.0f) continue;
        const float contrib = au * (w / mass);
        if (contrib < cutoff) continue;  // prune before it enters the frontier
        const NodeId v = ns[j];
        s.add(v, contrib);
        s.next.push_back(v);
        s.next_w.push_back(contrib);
      }
    }

    s.frontier.swap(s.next);
    s.front_w.swap(s.next_w);
  }

  // Rank the touched set. partial_sort because cap is normally far below the
  // number of nodes any two-hop walk reaches.
  std::vector<NodeId>& hits = s.touched;
  const std::size_t k = std::min(cap, hits.size());
  std::partial_sort(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(k), hits.end(),
                    [&](NodeId a, NodeId b) { return s.accum[a] > s.accum[b]; });

  std::size_t written = 0;
  for (std::size_t i = 0; i < k; ++i) {
    const float score = s.accum[hits[i]];
    if (score < cutoff) break;  // sorted, so everything after is smaller too
    out_nodes[written] = hits[i];
    out_scores[written] = score;
    ++written;
  }
  return written;
}

// ---------------------------------------------------------------------------
// walk_directional
//
// Steps between real memories rather than extrapolating a straight line.
// Straight-line extrapolation in embedding space leaves the data manifold
// almost immediately and lands nowhere meaningful; constraining the path to
// graph edges keeps every point real, and the path curves as a result. This is
// the Isomap idea: approximate a geodesic by traversing the neighbourhood
// graph.
//
// At each hop the next node is the neighbour whose displacement best matches
// the accumulated direction, and the direction is then blended toward the step
// actually taken. alpha is how much of the original heading is retained: 1.0
// holds a fixed bearing, 0.0 follows local curvature entirely.
// ---------------------------------------------------------------------------

std::size_t CsrSnapshot::walk_directional(NodeId start, const float* dir, std::uint32_t hops,
                                          float alpha, float cutoff, Tick tick, const float* emb,
                                          std::size_t emb_nodes, std::size_t dim,
                                          NodeId* out_path, std::size_t cap) const {
  if (!has(start) || emb == nullptr || dim == 0 || cap == 0) return 0;
  if (static_cast<std::size_t>(start) >= emb_nodes) return 0;

  std::vector<float> d(dir, dir + dim);
  const float dn = std::sqrt(dot(d.data(), d.data(), dim));
  if (dn <= 0.0f) return 0;
  for (float& x : d) x /= dn;

  std::vector<float> step(dim);
  std::vector<NodeId> visited;  // short by construction; linear scan is fine
  visited.reserve(hops + 1);

  NodeId cur = start;
  out_path[0] = cur;
  visited.push_back(cur);
  std::size_t written = 1;

  for (std::uint32_t h = 0; h < hops && written < cap; ++h) {
    const auto ns = row_nodes(cur);
    const auto vs = row_vals(cur);
    const float* pc = emb_row(emb, dim, cur);

    NodeId best = kInvalidNode;
    float best_cos = -2.0f;

    for (std::size_t j = 0; j < ns.size(); ++j) {
      const NodeId v = ns[j];
      if (static_cast<std::size_t>(v) >= emb_nodes) continue;
      if (decay_.decayed(vs[j], tick) < cutoff) continue;  // never step a dead edge
      if (std::find(visited.begin(), visited.end(), v) != visited.end()) continue;

      const float* pv = emb_row(emb, dim, v);
      float norm = 0.0f;
      for (std::size_t i = 0; i < dim; ++i) {
        step[i] = pv[i] - pc[i];
        norm += step[i] * step[i];
      }
      norm = std::sqrt(norm);
      if (norm <= 0.0f) continue;

      const float c = dot(step.data(), d.data(), dim) / norm;
      if (c > best_cos) {
        best_cos = c;
        best = v;
      }
    }

    if (best == kInvalidNode) break;  // dead end, or every neighbour visited

    // Blend the heading toward the step actually taken, so the path can curve
    // with the manifold instead of fighting it.
    const float* pb = emb_row(emb, dim, best);
    float norm = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
      step[i] = pb[i] - pc[i];
      norm += step[i] * step[i];
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
      float dn2 = 0.0f;
      for (std::size_t i = 0; i < dim; ++i) {
        d[i] = alpha * d[i] + (1.0f - alpha) * (step[i] / norm);
        dn2 += d[i] * d[i];
      }
      dn2 = std::sqrt(dn2);
      if (dn2 <= 0.0f) break;  // heading cancelled itself out
      for (float& x : d) x /= dn2;
    }

    cur = best;
    out_path[written++] = cur;
    visited.push_back(cur);
  }

  return written;
}

}  // namespace assoc
