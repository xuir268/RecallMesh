#include "assoc/csr.hpp"
#include "assoc/edge_store.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_set>
using namespace assoc;
using Edges=std::vector<std::pair<EdgeKey,EdgeVal>>;
void require(bool b){if(!b)throw std::runtime_error("CSR stress mismatch");}
void compare(const CsrSnapshot&a,const CsrSnapshot&b){
 require(a.node_count()==b.node_count() && a.edge_count()==b.edge_count());
 for(std::size_t i=0;i<a.node_count();++i){
  auto an=a.row_nodes(static_cast<NodeId>(i)),bn=b.row_nodes(static_cast<NodeId>(i));
  auto av=a.row_vals(static_cast<NodeId>(i)),bv=b.row_vals(static_cast<NodeId>(i));
  require(std::equal(an.begin(),an.end(),bn.begin(),bn.end()));
  require(std::equal(av.begin(),av.end(),bv.begin(),bv.end()));
  require(std::is_sorted(an.begin(),an.end()));
 }
}
Edges data(std::size_t count,std::size_t nodes,bool star,unsigned seed){
 std::mt19937 rng(seed);std::unordered_set<EdgeKey> seen;Edges edges;edges.reserve(count);
 while(edges.size()<count){NodeId a=star?1:1+rng()%nodes,b=1+rng()%nodes;if(a==b)continue;
  auto key=pack_key(a,b);if(seen.insert(key).second)edges.emplace_back(key,pack_val(float(1+rng()%30),rng()%100));}
 std::shuffle(edges.begin(),edges.end(),rng);return edges;
}
int main(int argc,char**argv){
 const bool bench=argc>1;
 compare(CsrSnapshot::build({},{}),CsrSnapshot::build_reference({},{}));
 for(unsigned seed=0;seed<100;++seed){auto e=data(10+seed*3,1000,false,seed);
  compare(CsrSnapshot::build(e,{}),CsrSnapshot::build_reference(e,{}));}
 for(auto count:{10000u,100000u,1000000u}){
  if(!bench && count>10000)continue;
  for(bool star:{false,true}){
   auto edges=data(count,star?count*2:count/5,star,42);
   compare(CsrSnapshot::build(edges,{}),CsrSnapshot::build_reference(edges,{}));
   std::vector<double> old,newer;
   for(int round=0;round<7;++round)for(int step=0;step<2;++step){
    bool optimized=(round+step)%2;auto begin=std::chrono::steady_clock::now();
    auto s=optimized?CsrSnapshot::build(edges,{}):CsrSnapshot::build_reference(edges,{});
    auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    require(s.edge_count()==count);if(round)(optimized?newer:old).push_back(ms);
   }
   std::sort(old.begin(),old.end());std::sort(newer.begin(),newer.end());
   std::cout<<"{\"edges\":"<<count<<",\"shape\":\""<<(star?"star":"uniform")<<"\",\"reference_ms\":"<<old[old.size()/2]<<",\"optimized_ms\":"<<newer[newer.size()/2]<<"}\n";
  }
 }
 // Readers retain old snapshots while writers reinforce and one publisher freezes.
 EdgeStore store(1<<16,Params{0,1,.001f});std::atomic<bool> done{false};
 std::thread publisher([&]{while(!done.load())store.freeze(0);});
 std::thread reader([&]{while(!done.load()){auto snap=store.snapshot();if(!snap)continue;
  auto n=snap->row_nodes(1);require(std::is_sorted(n.begin(),n.end()));}});
 std::vector<std::thread> writers;
 for(int t=0;t<4;++t)writers.emplace_back([&,t]{for(int i=0;i<100;++i){
  std::vector<NodeId>a(1000,1),b(1000,2+t);store.reinforce(a.data(),b.data(),a.size(),0);}});
 for(auto&t:writers)t.join();done.store(true);publisher.join();reader.join();store.freeze(0);
 for(NodeId i=2;i<6;++i)require(store.snapshot()->find_weight(1,i,0)==100000);
 std::cout<<"{\"regression\":\"passed\",\"concurrent_updates\":400000}\n";
}
