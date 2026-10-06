// Native shared-graph workload. Timing is observational, correctness is checked.
#include "assoc/edge_store.hpp"
#include "assoc/csr.hpp"
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
using namespace assoc;
using Clock=std::chrono::steady_clock;
struct Local { std::uint64_t reads=0; double read_ns=0,write_ns=0; std::vector<std::uint32_t> expected; };
int main(int argc,char** argv) {
 const unsigned iterations=argc>1?static_cast<unsigned>(std::stoul(argv[1])):20000;
 if(iterations==0 || iterations>200000) return 2;
 std::cout<<"agents,workload,repeat,write_ops,read_ops,seconds,write_ops_per_second,cas_retries,retries_per_write,sampled_write_ns,sampled_read_ns,freezes,correct,atomic_u64_lock_free\n";
 for(const char* workload:{"hot_edge","disjoint","uniform"}) for(unsigned count:{1u,2u,4u,8u,16u}) for(unsigned repeat=0;repeat<3;++repeat) {
  const bool hot=std::string(workload)=="hot_edge",disjoint=std::string(workload)=="disjoint";
  const std::size_t keys=hot?1:disjoint?count:4096;
  EdgeStore store(1<<16,Params{0,1,.001f});
  std::vector<NodeId>a(keys),b(keys);
  for(std::size_t k=0;k<keys;++k){a[k]=static_cast<NodeId>(2*k+1);b[k]=a[k]+1;}
  store.reinforce(a.data(),b.data(),keys,0);store.freeze(0);
  const auto before=store.stats(); std::vector<Local> local(count);
  for(auto& l:local)l.expected.resize(keys,0);
  std::barrier gate(static_cast<std::ptrdiff_t>(count+2));std::atomic<bool> done{false}; std::uint64_t freezes=0;
  std::thread publisher([&]{gate.arrive_and_wait();while(!done.load(std::memory_order_acquire)){
   store.freeze(0);++freezes;std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }});
  std::vector<std::thread> agents;
  for(unsigned n=0;n<count;++n)agents.emplace_back([&,n]{gate.arrive_and_wait();auto& l=local[n];
   for(unsigned i=0;i<iterations;++i){
    std::size_t k=hot?0:disjoint?n:(static_cast<std::size_t>(i)*2654435761u+n*17u)%keys;
    const bool sampled=(i%64)==0;auto t=sampled?Clock::now():Clock::time_point{};
    store.reinforce(&a[k],&b[k],1,0);++l.expected[k];
    if(sampled)l.write_ns+=std::chrono::duration<double,std::nano>(Clock::now()-t).count();
    if((i%4)==0){t=sampled?Clock::now():Clock::time_point{};
     auto snapshot=store.snapshot();NodeId out[4];float score[4];float w=1;
     const auto found=snapshot->activate(&a[k],&w,1,1,0,0,out,score,4);
     if(found!=2)std::terminate();++l.reads;
     if(sampled)l.read_ns+=std::chrono::duration<double,std::nano>(Clock::now()-t).count();
    }
   }
  });
  auto start=Clock::now();gate.arrive_and_wait();for(auto& t:agents)t.join();
  const double seconds=std::chrono::duration<double>(Clock::now()-start).count();done.store(true,std::memory_order_release);publisher.join();
  std::uint64_t reads=0;double wr=0,rd=0;bool correct=true;
  for(auto& l:local){reads+=l.reads;wr+=l.write_ns;rd+=l.read_ns;}
  for(std::size_t k=0;k<keys;++k){std::uint32_t expected=1;for(auto& l:local)expected+=l.expected[k];correct=correct&&(store.peek(a[k],b[k],0)==static_cast<float>(expected));}
  const auto after=store.stats();correct=correct&&after.rejected_full==0;
  const auto ops=static_cast<std::uint64_t>(count)*iterations,retries=after.cas_retries-before.cas_retries;
  const auto samples=static_cast<double>(count)*((iterations+63)/64);
  std::atomic<std::uint64_t> atom;
  std::cout<<count<<','<<workload<<','<<repeat<<','<<ops<<','<<reads<<','<<seconds<<','<<ops/seconds<<','<<retries<<','<<static_cast<double>(retries)/ops<<','<<wr/samples<<','<<rd/samples<<','<<freezes<<','<<correct<<','<<atom.is_lock_free()<<'\n';
  if(!correct)return 1;
 }
}
