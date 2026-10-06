#include "assoc/edge_store.hpp"
#include "assoc/csr.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>
int main(){
#if defined(__unix__) || defined(__APPLE__)
 using namespace assoc;
 auto dir=std::filesystem::temp_directory_path()/("recallmesh-segment-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 std::filesystem::create_directory(dir);
 try {
  NodeId a[]{1,1,2},b[]{2,3,4};EdgeStore before(128,Params{});
  before.reinforce(a,b,3,1);before.reinforce(a,b,3,7);before.freeze(9);
  auto path=(dir/"graph").string();before.save_checkpoint(path);EdgeStore after(128,Params{});after.restore_checkpoint(path);
  if(!after.snapshot()->mapped())throw std::runtime_error("not mapped");
  for(int i=0;i<3;++i)if(before.peek(a[i],b[i],9)!=after.peek(a[i],b[i],9))throw std::runtime_error("packed value recovery mismatch");
  before.reinforce(a,b,3,15);after.reinforce(a,b,3,15);
  for(int i=0;i<3;++i)if(before.peek(a[i],b[i],15)!=after.peek(a[i],b[i],15))throw std::runtime_error("future update mismatch");
  {std::ofstream f(path,std::ios::binary|std::ios::trunc);f<<"truncated";}
  bool rejected=false;try{EdgeStore invalid(128,Params{});invalid.restore_checkpoint(path);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected)throw std::runtime_error("corruption accepted");
  EdgeStore empty(128,Params{});empty.freeze(0);empty.save_checkpoint(path);EdgeStore restored(128,Params{});restored.restore_checkpoint(path);
  if(restored.snapshot()->edge_count()!=0)throw std::runtime_error("empty restore failed");
  std::filesystem::remove_all(dir);std::cout<<"Mapped CSR and exact delta recovery passed\n";
 }catch(...){std::filesystem::remove_all(dir);throw;}
#else
 std::cout<<"POSIX mmap unavailable: checkpoint test skipped\n";
#endif
}
