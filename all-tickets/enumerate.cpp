// g++ -O3 -std=c++17 -march=native -fopenmp enumerate.cpp -o enumerate.exe
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#include <immintrin.h>

using Word=std::uint64_t;
using Mask=std::uint16_t;
using Plane=std::vector<Word>;
struct Stage { Mask up,mid,down; };
static_assert(sizeof(Stage)==6);
constexpr int pairsPerMiddle=28, middlesPerUp=3003;
constexpr int perUp=middlesPerUp*pairsPerMiddle;
int bitcount(Word x){return __builtin_popcountll(x);}

// If a ticket's up + ordinary score is four, either down pick suffices.
// If it is three, both down picks must hit. These two sets are disjoint.
int rescued(const Plane& four,const Plane& three,const Plane& either,const Plane& both){
#if defined(__AVX512VPOPCNTDQ__)
    __m512i count=_mm512_setzero_si512();
    for(std::size_t j=0;j<four.size();j+=8){
        auto load=[j](const Plane& p){return _mm512_loadu_si512(p.data()+j);};
        const auto hits=_mm512_or_si512(_mm512_and_si512(load(four),load(either)),
                                        _mm512_and_si512(load(three),load(both)));
        count=_mm512_add_epi64(count,_mm512_popcnt_epi64(hits));
    }
    alignas(64) Word lanes[8];
    _mm512_storeu_si512(lanes,count);
    int total=0;
    for(Word n:lanes)total+=int(n);
    return total;
#else
    int total=0;
    for(std::size_t j=0;j<four.size();++j)total+=bitcount((four[j]&either[j])|(three[j]&both[j]));
    return total;
#endif
}

int main(int argc,char** argv){
    if(argc!=3){std::cerr<<"Usage: enumerate stage.bin counts.bin\n";return 2;}
    std::ifstream in(argv[1],std::ios::binary|std::ios::ate);
    if(!in)throw std::runtime_error("input");
    const auto bytes=static_cast<std::streamoff>(in.tellg());
    if(bytes<=0 || bytes%6)throw std::runtime_error("Invalid trial file");
    const int n=int(bytes/6);
    const int words=((n+511)/512)*8;
    in.seekg(0);
    std::array<std::array<Plane,16>,3> scores;
    for(auto& category:scores)for(auto& p:category)p.assign(words,0);
    for(int i=0;i<n;i++){
        Stage r{};in.read(reinterpret_cast<char*>(&r),sizeof(r));
        if(!in||bitcount(r.up)!=2||bitcount(r.mid)!=6||bitcount(r.down)!=2||
           (r.up&r.mid)||(r.up&r.down)||(r.mid&r.down))throw std::runtime_error("Invalid stage");
        Mask masks[]={r.up,r.mid,r.down};
        for(int k=0;k<3;k++)for(unsigned int m=masks[k];m;m&=m-1)
            scores[k][__builtin_ctz(m)][i/64]|=Word(1)<<(i%64);
    }
    std::vector<Mask> pairs,middles;
    for(unsigned m=1;m<65536;m++){
        if(bitcount(m)==2)pairs.push_back(Mask(m));
        if(bitcount(m)==6)middles.push_back(Mask(m));
    }
    std::vector<std::vector<short>> middleRank(pairs.size(),std::vector<short>(65536,-1));
    std::vector<Plane> downAny,downBoth;
    for(std::size_t u=0;u<pairs.size();u++){
        int k=0;for(auto m:middles)if(!(m&pairs[u]))middleRank[u][m]=short(k++);
        if(k!=middlesPerUp)throw std::runtime_error("middle count");
        int a=__builtin_ctz(pairs[u]),b=__builtin_ctz(pairs[u]&(pairs[u]-1));
        Plane any(words),both(words);
        for(int j=0;j<words;j++){any[j]=scores[2][a][j]|scores[2][b][j];both[j]=scores[2][a][j]&scores[2][b][j];}
        downAny.push_back(std::move(any));downBoth.push_back(std::move(both));
    }
    const std::uint32_t empty=UINT32_MAX;
    std::vector<std::uint32_t> wins(pairs.size()*perUp,empty);
    omp_set_num_threads(std::min(12,omp_get_max_threads()));
    std::cout<<"Trials "<<n<<", tickets "<<wins.size()<<", threads "<<omp_get_max_threads()<<std::endl;
    #pragma omp parallel
    {
        Plane one(words),two(words),three(words),four(words),fivePlus(words),needOne(words),needTwo(words);
        #pragma omp for schedule(dynamic,8)
        for(int mi=0;mi<int(middles.size());mi++){
            const Mask mid=middles[mi];
            int automatic=0;
            for(int j=0;j<words;j++){
                Word o=0,t=0,f=0;
                for(unsigned m=mid;m;m&=m-1){
                    Word val=scores[1][__builtin_ctz(m)][j];
                    Word carry=o&val; o^=val; f|=t&carry; t^=carry;
                }
                one[j]=o&~t&~f; two[j]=~o&t&~f; three[j]=o&t&~f;
                four[j]=~o&~t&f; fivePlus[j]=f&(o|t); automatic+=bitcount(fivePlus[j]);
            }
            for(int u=0;u<int(pairs.size());u++){
                const Mask up=pairs[u]; if(up&mid)continue;
                int a=__builtin_ctz(up),b=__builtin_ctz(up&(up-1)),base=automatic;
                for(int j=0;j<words;j++){
                    Word x=scores[0][a][j],y=scores[0][b][j];
                    Word both=x&y,either=x|y,single=x^y;
                    base+=bitcount((four[j]&either)|(three[j]&both));
                    needOne[j]=(four[j]&~either)|(three[j]&single)|(two[j]&both);
                    needTwo[j]=(three[j]&~either)|(two[j]&single)|(one[j]&both);
                }
                int index=u*perUp+middleRank[u][mid]*pairsPerMiddle;
                for(int d=0;d<int(pairs.size());d++)if(!(pairs[d]&(up|mid)))
                    wins[index++]=base+rescued(needOne,needTwo,downAny[d],downBoth[d]);
            }
        }
    }
    for(auto v:wins)if(v>static_cast<unsigned>(n))throw std::runtime_error("Missing ticket or invalid count");
    std::ofstream out(argv[2],std::ios::binary);
    out.write(reinterpret_cast<const char*>(wins.data()),wins.size()*sizeof(wins[0]));
    if(!out)throw std::runtime_error("Output failed");
    std::cout<<"All tickets scored."<<std::endl;
}
