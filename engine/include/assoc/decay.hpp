#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <stdexcept>

#include "assoc/types.hpp"

namespace assoc {

// ---------------------------------------------------------------------------
// DecayTable
//
// Weights are never swept. Every read applies decay as a pure function of the
// elapsed tick count:
//
//     w_effective = w * (1 - lambda) ^ (now - last_tick)
//
// Recent-age factors are precomputed up to a bounded cache size. Older ages
// use pow() rather than forcibly erasing a potentially strong edge. The floor
// applies to the effective weight when freeze() decides which edges to retain.
//
// Immutable after construction, so concurrent readers need no synchronisation.
// ---------------------------------------------------------------------------

class DecayTable {
 public:
  // Hard ceiling so a very small lambda cannot allocate without bound.
  static constexpr std::size_t kMaxEntries = 1u << 16;

  DecayTable() : DecayTable(Params{}) {}

  explicit DecayTable(Params p) : params_(p) {
    if (!std::isfinite(p.lambda) || !std::isfinite(p.eta) || p.eta < 0 ||
        !std::isfinite(p.floor) || p.floor <= 0 || p.floor >= 1)
      throw std::invalid_argument("finite lambda, nonnegative eta, and 0 < floor < 1 required");
    const double base = 1.0 - static_cast<double>(p.lambda);

    if (base >= 1.0) {
      // lambda <= 0: decay disabled. The table is unused; factor is always 1.
      no_decay_ = true;
      return;
    }
    if (base <= 0.0) {
      // lambda >= 1: anything not touched this very tick is gone.
      table_.assign(1, 1.0f);
      return;
    }

    // Entries needed before the factor falls under the floor.
    const double floor = std::max(static_cast<double>(p.floor), 1e-12);
    const double n = std::log(floor) / std::log(base);
    const std::size_t count =
        n >= static_cast<double>(kMaxEntries - 1) ? kMaxEntries
        : static_cast<std::size_t>(std::ceil(n)) + 1;

    table_.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      // pow() per entry rather than a running product: built once, and the
      // recurrence would accumulate error across thousands of steps.
      table_[i] = static_cast<float>(std::pow(base, static_cast<double>(i)));
    }
  }

  // Cached multiplier for recent ages; older ages use a power fallback.
  [[nodiscard]] float factor(Tick elapsed) const noexcept {
    if (no_decay_) return 1.0f;
    const std::size_t e = static_cast<std::size_t>(elapsed);
    return e < table_.size() ? table_[e]
        : static_cast<float>(std::pow(std::max(0.0, 1.0 - static_cast<double>(params_.lambda)),
                                      static_cast<double>(elapsed)));
  }

  // Elapsed ticks, clamped. A snapshot read can legitimately carry a tick
  // older than an edge written concurrently, and unsigned subtraction would
  // turn that into a huge age and silently zero a live edge.
  [[nodiscard]] static Tick elapsed(Tick now, Tick last) noexcept {
    return now > last ? static_cast<Tick>(now - last) : Tick{0};
  }

  // Decayed weight straight from a packed EdgeVal. This is the hot call:
  // every adjacency walk runs it once per edge visited.
  [[nodiscard]] float decayed(EdgeVal v, Tick now) const noexcept {
    return val_weight(v) * factor(elapsed(now, val_tick(v)));
  }

  // Hebbian update. Decay to the present, then reinforce. Used by the CAS
  // loop in EdgeStore, so it stays allocation-free and branch-light.
  [[nodiscard]] EdgeVal reinforce(EdgeVal v, Tick now, float amount) const noexcept {
    const Tick effective_now = std::max(now, val_tick(v));
    return pack_val(decayed(v, effective_now) + amount, effective_now);
  }

  // True once a weight is small enough to drop at the next freeze.
  [[nodiscard]] bool is_dead(float w) const noexcept { return w < params_.floor; }

  [[nodiscard]] const Params& params() const noexcept { return params_; }

  // Cached age range (not a hard expiration horizon).
  [[nodiscard]] std::size_t horizon() const noexcept {
    return no_decay_ ? kMaxEntries : table_.size();
  }

 private:
  Params params_{};
  std::vector<float> table_{};
  bool no_decay_ = false;
};

}  // namespace assoc
