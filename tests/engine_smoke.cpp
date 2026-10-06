#include "assoc/edge_store.hpp"
#include "assoc/csr.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
void check(bool condition) { if (!condition) throw std::runtime_error("check failed"); }
int main() {
  using namespace assoc;
  EdgeStore store(128, Params{0.5f, 1.0f, 0.001f});
  NodeId a[]{1,1,2}, b[]{2,3,4};
  store.reinforce(a,b,3,10);
  store.freeze(10);
  check(store.snapshot()->edge_count()==3);
  check(store.snapshot()->find_weight(1,2,11)==0.5f);
  NodeId seeds[]{1}, nodes[8]; float seed_w[]{1}, scores[8];
  check(store.activate(seeds,seed_w,1,1,0,10,nodes,scores,8)==3);
  check(nodes[0]==1 && scores[0]==1);
  float emb[]{0,0, 0,0, 1,0, 0,1, 2,0};
  store.attach_embeddings(emb,5,2);
  float dir[]{1,0}; NodeId path[8];
  check(store.walk_directional(1,dir,2,1,10,path,8)==3);
  check(path[0]==1 && path[1]==2 && path[2]==4);
  EdgeStore concurrent(128,Params{0,1,0.001f});
  std::vector<std::thread> threads;
  for(int i=0;i<8;++i) threads.emplace_back([&]{
    std::vector<NodeId> src(50000,1), dst(50000,2);
    concurrent.reinforce(src.data(),dst.data(),src.size(),0);
  });
  for(auto& t:threads)t.join();
  check(concurrent.peek(1,2,0)==400000);
  std::cout << "C++ queries and 400000 concurrent reinforcements passed\n";
}
