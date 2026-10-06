#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>

#include "assoc/csr.hpp"
#include "assoc/edge_store.hpp"

namespace nb = nanobind;
using namespace assoc;
using namespace nb::literals;

namespace {

// Zero-copy, contiguous, correctly typed views. nanobind rejects anything
// else at the call boundary rather than silently copying.
using U32In = nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F32In = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F32In2 = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Hand a heap vector to Python without copying: the capsule owns it and frees
// it when the array is collected.
template <typename T>
nb::ndarray<nb::numpy, T> steal(std::vector<T>&& v, std::size_t n) {
  auto* held = new std::vector<T>(std::move(v));
  held->resize(n);
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T>(held->data(), {n}, owner);
}

}  // namespace

NB_MODULE(_engine, m) {
  m.doc() = "Associative memory graph engine";

  nb::class_<EdgeStore>(m, "EdgeStore")
      .def(
          "__init__",
          [](EdgeStore* self, std::size_t capacity, float lambda, float eta, float floor) {
            // Mask arithmetic requires a power of two; round up rather than
            // trusting the caller.
            if (capacity > (std::size_t{1} << (std::numeric_limits<std::size_t>::digits - 1)))
              throw std::invalid_argument("capacity too large");
            std::size_t cap = 2;
            while (cap < capacity) cap <<= 1;
            new (self) EdgeStore(cap, Params{lambda, eta, floor});
          },
          "capacity"_a, "lambda_"_a = 0.05f, "eta"_a = 1.0f, "floor"_a = 1e-3f)

      // -- write path ---------------------------------------------------------
      // call_guard releases the GIL for the whole call. Safe here because
      // nothing touches a Python object: the arrays are borrowed views and the
      // return is void. Without this the lock-free engine would serialise
      // behind the interpreter lock and the entire design would be pointless.
      .def(
          "reinforce",
          [](EdgeStore& s, U32In src, U32In dst, Tick tick) {
            if (src.shape(0) != dst.shape(0)) throw std::invalid_argument("src and dst lengths differ");
            const std::size_t n = src.shape(0);
            s.reinforce(src.data(), dst.data(), n, tick);
          },
          "src"_a, "dst"_a, "tick"_a, nb::call_guard<nb::gil_scoped_release>())

      .def("save_checkpoint", &EdgeStore::save_checkpoint, "path"_a, nb::call_guard<nb::gil_scoped_release>())
      .def("restore_checkpoint", &EdgeStore::restore_checkpoint, "path"_a, nb::call_guard<nb::gil_scoped_release>())
      .def("freeze", &EdgeStore::freeze, "tick"_a, nb::call_guard<nb::gil_scoped_release>())

      // -- queries ------------------------------------------------------------
      // These allocate a numpy array to return, which touches the interpreter,
      // so the GIL is released around the compute only and reacquired before
      // the result is built. call_guard would be wrong here.
      .def(
          "activate",
          [](const EdgeStore& s, U32In seeds, F32In seed_w, std::uint32_t hops, float cutoff,
             Tick tick, std::size_t cap) {
            if (seeds.shape(0) != seed_w.shape(0)) throw std::invalid_argument("seed lengths differ");
            std::vector<NodeId> nodes(cap);
            std::vector<float> scores(cap);
            std::size_t n = 0;
            {
              nb::gil_scoped_release unlock;
              n = s.activate(seeds.data(), seed_w.data(), seeds.shape(0), hops, cutoff, tick,
                             nodes.data(), scores.data(), cap);
            }
            return nb::make_tuple(steal(std::move(nodes), n), steal(std::move(scores), n));
          },
          "seeds"_a, "seed_w"_a, "hops"_a = 2, "cutoff"_a = 1e-3f, "tick"_a = 0, "cap"_a = 64)

      .def("activate_profile",
          [](const EdgeStore& store, U32In seeds, F32In seed_w, std::uint32_t hops,
             float cutoff, Tick tick, std::size_t cap, bool optimized, bool trace) {
            if (seeds.shape(0) != seed_w.shape(0)) throw std::invalid_argument("seed lengths differ");
            std::vector<NodeId> nodes(cap); std::vector<float> scores(cap);
            ActivationProfile p; p.trace = trace;
            std::size_t n = 0;
            {
              nb::gil_scoped_release unlock;
              const auto snap = store.snapshot();
              if (snap) n = snap->activate(seeds.data(), seed_w.data(), seeds.shape(0),
                  hops, cutoff, tick, nodes.data(), scores.data(), cap, &p, optimized);
            }
            nb::dict metrics;
            metrics["row_visits"] = p.row_visits;
            metrics["unique_rows"] = p.unique_rows;
            metrics["edge_visits"] = p.edge_visits;
            metrics["decay_evaluations"] = p.decay_evaluations;
            metrics["merged_entries"] = p.merged_entries;
            metrics["frontier_entries"] = p.frontier_entries;
            metrics["max_frontier"] = p.max_frontier;
            nb::list transitions;
            for (const auto& t : p.transitions)
              transitions.append(nb::make_tuple(t.source,t.target,t.hop,t.contribution));
            return nb::make_tuple(steal(std::move(nodes),n), steal(std::move(scores),n), metrics, transitions);
          }, "seeds"_a, "seed_w"_a, "hops"_a=2, "cutoff"_a=1e-3f,
          "tick"_a=0, "cap"_a=64, "optimized"_a=true, "trace"_a=false)

      .def(
          "weights_for_pairs",
          [](const EdgeStore& s, U32In a, U32In b, Tick tick) {
            if (a.shape(0) != b.shape(0)) throw std::invalid_argument("pair lengths differ");
            const std::size_t n = a.shape(0);
            std::vector<float> out(n);
            {
              nb::gil_scoped_release unlock;
              s.weights_for_pairs(a.data(), b.data(), n, tick, out.data());
            }
            return steal(std::move(out), n);
          },
          "a"_a, "b"_a, "tick"_a = 0)

      .def(
          "edge_mass_along",
          [](const EdgeStore& s, U32In ordered, Tick tick) {
            return s.edge_mass_along(ordered.data(), ordered.shape(0), tick);
          },
          "ordered"_a, "tick"_a = 0, nb::call_guard<nb::gil_scoped_release>())

      .def(
          "neighbors",
          [](const EdgeStore& s, NodeId node, float cutoff, Tick tick, std::size_t cap) {
            std::vector<NodeId> nodes(cap);
            std::vector<float> w(cap);
            std::size_t n = 0;
            {
              nb::gil_scoped_release unlock;
              n = s.neighbors(node, cutoff, tick, nodes.data(), w.data(), cap);
            }
            return nb::make_tuple(steal(std::move(nodes), n), steal(std::move(w), n));
          },
          "node"_a, "cutoff"_a = 1e-3f, "tick"_a = 0, "cap"_a = 256)

      .def(
          "walk_directional",
          [](const EdgeStore& s, NodeId start, F32In dir, std::uint32_t hops, float alpha,
             Tick tick, std::size_t cap) {
            if (dir.shape(0) != s.embedding_dim()) throw std::invalid_argument("direction dimension mismatch");
            std::vector<NodeId> path(cap);
            std::size_t n = 0;
            {
              nb::gil_scoped_release unlock;
              n = s.walk_directional(start, dir.data(), hops, alpha, tick, path.data(), cap);
            }
            return steal(std::move(path), n);
          },
          "start"_a, "dir"_a, "hops"_a = 4, "alpha"_a = 0.5f, "tick"_a = 0, "cap"_a = 32)

      // -- embeddings ---------------------------------------------------------
      // Borrowed, never owned. keep_alive ties the array's lifetime to the
      // store so Python cannot free it out from under a walk.
      .def(
          "attach_embeddings",
          [](EdgeStore& s, F32In2 emb) {
            s.attach_embeddings(emb.data(), emb.shape(0), emb.shape(1));
          },
          "emb"_a, nb::keep_alive<1, 2>())

      // -- introspection ------------------------------------------------------
      .def("peek", &EdgeStore::peek, "a"_a, "b"_a, "tick"_a = 0)

      .def("stats",
           [](const EdgeStore& s) {
             const Stats st = s.stats();
             nb::dict d;
             d["capacity"] = st.capacity;
             d["live_edges"] = st.live_edges;
             d["inserts"] = st.inserts;
             d["updates"] = st.updates;
             d["cas_retries"] = st.cas_retries;
             d["rejected_full"] = st.rejected_full;
             d["dropped_edges"] = st.dropped_edges;
             d["load_factor"] = st.load_factor();
             d["mean_probe"] = st.mean_probe();
             return d;
           })

      .def_prop_ro("edge_count", [](const EdgeStore& s) {
        const auto snap = s.snapshot();
        return snap ? snap->edge_count() : std::size_t{0};
      })

      .def_prop_ro("node_count", [](const EdgeStore& s) {
        const auto snap = s.snapshot();
        return snap ? snap->node_count() : std::size_t{0};
      });
}
