#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace assoc {

// ---------------------------------------------------------------------------
// Identifiers
//
// The engine knows nothing about text, embeddings, or models. A memory is an
// opaque 32-bit id assigned by the Python node table. This keeps the engine
// small enough to reason about and lets its tests run without a corpus.
// ---------------------------------------------------------------------------

using NodeId = std::uint32_t;
using Tick = std::uint32_t;

// Reserved: node 0 never exists, so a zero key marks an empty hash slot.
inline constexpr NodeId kInvalidNode = 0;

inline constexpr NodeId kMaxNode = std::numeric_limits<NodeId>::max();

// ---------------------------------------------------------------------------
// Edge key
//
// An edge is packed into a single u64 so a slot probe is one load and one
// compare. Associations are symmetric (co-activation has no direction), so the
// endpoints are ordered before packing: (a,b) and (b,a) map to one slot.
// ---------------------------------------------------------------------------

using EdgeKey = std::uint64_t;

inline constexpr EdgeKey kEmptyKey = 0;

[[nodiscard]] inline constexpr EdgeKey pack_key(NodeId a, NodeId b) noexcept {
  const NodeId lo = a < b ? a : b;
  const NodeId hi = a < b ? b : a;
  return (static_cast<EdgeKey>(lo) << 32) | static_cast<EdgeKey>(hi);
}

[[nodiscard]] inline constexpr NodeId key_lo(EdgeKey k) noexcept {
  return static_cast<NodeId>(k >> 32);
}

[[nodiscard]] inline constexpr NodeId key_hi(EdgeKey k) noexcept {
  return static_cast<NodeId>(k & 0xFFFFFFFFULL);
}

// Given one endpoint, return the other. Callers walking an adjacency row know
// which endpoint they started from.
[[nodiscard]] inline constexpr NodeId key_other(EdgeKey k, NodeId from) noexcept {
  const NodeId lo = key_lo(k);
  return lo == from ? key_hi(k) : lo;
}

// ---------------------------------------------------------------------------
// Edge value
//
// Weight and last-touched tick share one u64 so reinforcement is a single CAS:
// no lock, no torn read between the two fields, no reclamation problem. The
// pairing matters because decay is a pure function of (weight, tick) and would
// be wrong if the two could be observed out of step.
//
//   bits 63..32  f32 weight, bit-cast
//   bits 31..0   u32 tick
// ---------------------------------------------------------------------------

using EdgeVal = std::uint64_t;

[[nodiscard]] inline EdgeVal pack_val(float weight, Tick tick) noexcept {
  return (static_cast<EdgeVal>(std::bit_cast<std::uint32_t>(weight)) << 32) |
         static_cast<EdgeVal>(tick);
}

[[nodiscard]] inline float val_weight(EdgeVal v) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(v >> 32));
}

[[nodiscard]] inline constexpr Tick val_tick(EdgeVal v) noexcept {
  return static_cast<Tick>(v & 0xFFFFFFFFULL);
}

// ---------------------------------------------------------------------------
// Learning parameters
//
// lambda: per-tick decay rate. w *= (1 - lambda) ^ elapsed.
// eta:    reinforcement added per co-activation.
// floor:  weights below this are treated as zero and dropped at freeze.
// ---------------------------------------------------------------------------

struct Params {
  float lambda = 0.05f;
  float eta = 1.0f;
  float floor = 1e-3f;
};

// ---------------------------------------------------------------------------
// Observability
//
// Contention is the thing this design is meant to survive, so CAS retries are
// a first-class metric rather than a debug counter. If cas_retries climbs with
// writer count, the benchmark has found something worth reporting.
// ---------------------------------------------------------------------------

struct Stats {
  std::size_t capacity = 0;
  std::size_t live_edges = 0;
  std::uint64_t inserts = 0;
  std::uint64_t updates = 0;
  std::uint64_t cas_retries = 0;
  std::uint64_t probe_length_sum = 0;
  std::uint64_t rejected_full = 0;
  std::uint64_t dropped_edges = 0;

  [[nodiscard]] double load_factor() const noexcept {
    return capacity ? static_cast<double>(live_edges) / static_cast<double>(capacity) : 0.0;
  }

  [[nodiscard]] double mean_probe() const noexcept {
    const std::uint64_t ops = inserts + updates;
    return ops ? static_cast<double>(probe_length_sum) / static_cast<double>(ops) : 0.0;
  }
};

}  // namespace assoc
