#include "assoc/csr.hpp"
#include "assoc/edge_store.hpp"
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace assoc {
namespace {
using Header=std::array<std::uint64_t,11>;
constexpr std::uint64_t magic=0x524d435352303031ULL;
std::uint64_t checksum(const unsigned char* p,std::size_t n){std::uint64_t h=14695981039346656037ULL;for(std::size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;}
std::size_t align8(std::size_t n){return (n+7)&~std::size_t{7};}
Header layout(std::size_t nodes,std::size_t edges,Params p){
 if(nodes>std::numeric_limits<NodeId>::max() || edges>(1ULL<<32))throw std::runtime_error("segment too large");
 const auto entry=edges*2,nb=sizeof(Header)+(nodes+1)*sizeof(std::uint64_t),val=align8(nb+entry*sizeof(NodeId));
 return {magic,1,nodes,edges,nb,val,val+entry*sizeof(EdgeVal),std::bit_cast<std::uint32_t>(p.lambda),std::bit_cast<std::uint32_t>(p.eta),std::bit_cast<std::uint32_t>(p.floor),0};
}
}
void CsrSnapshot::save_segment(const std::string& path) const {
 static_assert(sizeof(std::size_t)==8); // version 1 is a 64-bit little-endian format.
 if constexpr(std::endian::native!=std::endian::little)throw std::runtime_error("segment endian unsupported");
 auto header=layout(n_nodes_,n_edges_,params_);std::vector<unsigned char> bytes(header[6],0);
 auto offsets=offset_view();auto nodes=node_view();auto values=value_view();
 std::memcpy(bytes.data()+sizeof(Header),offsets.data(),offsets.size_bytes());
 if(!nodes.empty())std::memcpy(bytes.data()+header[4],nodes.data(),nodes.size_bytes());
 if(!values.empty())std::memcpy(bytes.data()+header[5],values.data(),values.size_bytes());
 header[10]=checksum(bytes.data()+sizeof(Header),bytes.size()-sizeof(Header));std::memcpy(bytes.data(),header.data(),sizeof(Header));
 std::ofstream file(path,std::ios::binary|std::ios::trunc);file.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));file.flush();
 if(!file)throw std::runtime_error("cannot write CSR segment");
}
CsrSnapshot CsrSnapshot::map_segment(const std::string& path,Params expected) {
#if defined(__unix__) || defined(__APPLE__)
 if constexpr(std::endian::native!=std::endian::little)throw std::runtime_error("segment endian unsupported");
 int fd=::open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("cannot open CSR segment");
 struct stat st{};if(::fstat(fd,&st)!=0 || st.st_size<static_cast<off_t>(sizeof(Header))){::close(fd);throw std::runtime_error("truncated CSR segment");}
 const auto length=static_cast<std::size_t>(st.st_size);void* ptr=::mmap(nullptr,length,PROT_READ,MAP_PRIVATE,fd,0);::close(fd);
 if(ptr==MAP_FAILED)throw std::runtime_error("cannot mmap CSR segment");
 std::shared_ptr<void> owner(ptr,[length](void* p){::munmap(p,length);});
 Header h{};std::memcpy(h.data(),ptr,sizeof(Header));
 if(h[0]!=magic || h[1]!=1)throw std::runtime_error("unsupported CSR segment");
 auto wanted=layout(h[2],h[3],expected);
 for(std::size_t i=0;i<10;++i)if(h[i]!=wanted[i])throw std::runtime_error("CSR metadata mismatch");
 if(h[6]!=length)throw std::runtime_error("CSR length mismatch");
 const auto* raw=static_cast<const unsigned char*>(ptr);
 if(h[10]!=checksum(raw+sizeof(Header),length-sizeof(Header)))throw std::runtime_error("CSR checksum mismatch");
 CsrSnapshot s(expected);s.mapping_=std::move(owner);s.n_nodes_=h[2];s.n_edges_=h[3];
 s.mapped_offsets_={reinterpret_cast<const std::size_t*>(raw+sizeof(Header)),s.n_nodes_+1};
 s.mapped_nodes_={reinterpret_cast<const NodeId*>(raw+h[4]),s.n_edges_*2};
 s.mapped_values_={reinterpret_cast<const EdgeVal*>(raw+h[5]),s.n_edges_*2};
 auto offsets=s.offset_view();if(offsets[0]!=0 || offsets.back()!=s.n_edges_*2)throw std::runtime_error("CSR boundaries invalid");
 for(std::size_t u=0;u<s.n_nodes_;++u){
  if(offsets[u]>offsets[u+1] || offsets[u+1]>s.n_edges_*2)throw std::runtime_error("CSR offsets invalid");
  NodeId prior=0;bool first=true;
  auto ns=s.row_nodes(static_cast<NodeId>(u));auto vs=s.row_vals(static_cast<NodeId>(u));
  for(std::size_t j=0;j<ns.size();++j){
   if(ns[j]>=s.n_nodes_ || ns[j]==u || (!first && ns[j]<=prior) || !std::isfinite(val_weight(vs[j])) || val_weight(vs[j])<0)throw std::runtime_error("CSR row invalid");
   prior=ns[j];first=false;
  }
 }
 // Verify reverse entries carry the same packed value before trusting recovery.
 for(std::size_t u=0;u<s.n_nodes_;++u){auto ns=s.row_nodes(static_cast<NodeId>(u));auto vs=s.row_vals(static_cast<NodeId>(u));
  for(std::size_t j=0;j<ns.size();++j){auto reverse=s.row_nodes(ns[j]);auto it=std::lower_bound(reverse.begin(),reverse.end(),static_cast<NodeId>(u));
   if(it==reverse.end() || *it!=u || s.row_vals(ns[j])[it-reverse.begin()]!=vs[j])throw std::runtime_error("CSR symmetry invalid");
  }
 }
 return s;
#else
 (void)path;(void)expected;throw std::runtime_error("mmap checkpoints require POSIX on this build");
#endif
}
void EdgeStore::save_checkpoint(const std::string& path) const {
 auto snap=snapshot();if(!snap)throw std::runtime_error("freeze before checkpoint");
 std::vector<std::pair<EdgeKey,EdgeVal>> edges;
 for(std::size_t i=0;i<capacity_;++i){auto k=slots_[i].key.load(std::memory_order_acquire);if(k!=kEmptyKey)edges.emplace_back(k,slots_[i].val.load(std::memory_order_acquire));}
 CsrSnapshot::build(std::move(edges),decay_.params()).save_segment(path+".delta");snap->save_segment(path);
}
void EdgeStore::restore_checkpoint(const std::string& path) {
 if(live_.load()!=0)throw std::runtime_error("restore requires a fresh engine without concurrent writers");
 auto delta=CsrSnapshot::map_segment(path+".delta",decay_.params());
 auto cold=std::make_shared<const CsrSnapshot>(CsrSnapshot::map_segment(path,decay_.params()));
 if(delta.edge_count()>max_live())throw std::runtime_error("checkpoint exceeds graph capacity");
 for(std::size_t u=0;u<delta.node_count();++u){auto ns=delta.row_nodes(static_cast<NodeId>(u));auto vs=delta.row_vals(static_cast<NodeId>(u));
  for(std::size_t j=0;j<ns.size();++j)if(u<ns[j]){auto key=pack_key(static_cast<NodeId>(u),ns[j]);auto idx=slot_of(key);std::size_t probe=0;
   while(slots_[idx].key.load()!=kEmptyKey && probe<kMaxProbe){idx=(idx+1)&mask_;++probe;}
   if(probe>=kMaxProbe)throw std::runtime_error("checkpoint probe limit");
   slots_[idx].val.store(vs[j]);slots_[idx].key.store(key);live_.fetch_add(1);
  }
 }
 std::atomic_store_explicit(&snapshot_,std::move(cold),std::memory_order_release);
}
}
