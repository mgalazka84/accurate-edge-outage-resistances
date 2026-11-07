// Kernel-only C ABI for the edge recursion (2), vertex recursion (3),
// and independent subtraction-free LDL solves (4). Inputs are resistances.
// Compile without fast-math or contraction; see vertex_study.py.
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
using Vec=std::vector<double>;
using Ids=std::vector<int>;
struct Matrix {
    int n; Vec a;
    explicit Matrix(int n=0): n(n),a(size_t(n)*n) {}
    double& at(int i,int j){return a[size_t(i)*n+j];}
    double at(int i,int j) const{return a[size_t(i)*n+j];}
    void set(int i,int j,double x){at(i,j)=at(j,i)=x;}
};
enum Metric {OPS,SUBNORMAL,ZERO_POSITIVE,NONFINITE,HW_UNDERFLOW,NODES,LEAVES,COUNT};
struct PathStats {
    int degree_cost=0, pivots=0, levels=0, max_degree=0;
    void eliminate(int degree) {
        int height=0;
        for(int size=1;size<degree;size*=2)++height;
        degree_cost+=3+height;
        ++pivots;
        max_degree=std::max(max_degree,degree);
    }
};
template<bool audit> struct Solver {
    int v,k; const int *a,*b; const double* r; double* out; std::uint64_t* counts;
    int* path_data=nullptr;
    Vec weight,exclusive; Matrix pairs;
    std::vector<Ids> groups; std::vector<char> done;
    double record(double x,bool positive) {
        if constexpr(audit) {
            ++counts[OPS];
            if(std::fpclassify(x)==FP_SUBNORMAL)++counts[SUBNORMAL];
            if(x==0 && positive)++counts[ZERO_POSITIVE];
            if(!std::isfinite(x))++counts[NONFINITE];
        }
        if(!std::isfinite(x))throw std::runtime_error("nonfinite arithmetic");
        return x;
    }
    double add(double x,double y){return record(x+y,x>0 || y>0);}
    double mul(double x,double y){return record(x*y,x>0 && y>0);}
    double div(double x,double y){if(y<=0)throw std::runtime_error("nonpositive divisor");return record(x/y,x>0);}
    size_t key(int x,int y) const {return size_t(std::min(x,y))*v+std::max(x,y);}
    double sum(Vec x) {
        while(x.size()>1){
            size_t n=x.size(),j=0;
            for(size_t i=0;i<n;i+=2)x[j++]=i+1<n?add(x[i],x[i+1]):x[i];
            x.resize(j);
        }
        return x.empty()?0:x[0];
    }
    void assemble_pairs() {
        for(auto& group:groups) {
            if(group.empty())continue;
            size_t n=1;while(n<group.size())n*=2;
            Vec tree(2*n),outside(2*n);
            for(size_t j=0;j<group.size();++j)tree[n+j]=weight[group[j]];
            for(size_t j=n-1;j>0;--j)tree[j]=add(tree[2*j],tree[2*j+1]);
            for(size_t j=1;j<n;++j){
                outside[2*j]=add(outside[j],tree[2*j+1]);
                outside[2*j+1]=add(outside[j],tree[2*j]);
            }
            for(size_t j=0;j<group.size();++j)exclusive[group[j]]=outside[n+j];
            int e=group[0];pairs.set(a[e],b[e],tree[1]);
        }
    }
    // Eliminate real nonterminals in their current order. Padding slots are labels -1.
    Matrix reduce(Matrix& g,const Ids& slots,const Ids& keep, PathStats* path=nullptr) {
        std::vector<char> terminal(g.n),active(g.n);
        for(int i:keep)terminal[i]=1;
        for(int i=0;i<g.n;++i)active[i]=(slots[i]>=0);
        for(int h=0;h<g.n;++h) {
            if(!active[h] || terminal[h])continue;
            Ids neighbors;Vec row;
            for(int i=0;i<g.n;++i)if(active[i] && i!=h && g.at(h,i)>0) {
                neighbors.push_back(i);row.push_back(g.at(h,i));
            }
            if(row.empty())throw std::runtime_error("zero pivot");
            if(path)path->eliminate(static_cast<int>(row.size()));
            double pivot=sum(row);
            for(size_t j=1;j<neighbors.size();++j){
                const int y=neighbors[j];const double ratio=div(row[j],pivot);
                for(size_t i=0;i<j;++i){
                    int x=neighbors[i];g.set(x,y,add(g.at(x,y),mul(row[i],ratio)));
                }
            }
            active[h]=0;
        }
        Matrix small(keep.size());
        for(int j=0;j<small.n;++j)for(int i=0;i<j;++i)small.set(i,j,g.at(keep[i],keep[j]));
        return small;
    }
    void vertex_visit(const Ids& slots,const Matrix& bg, PathStats path={}) {
        if constexpr(audit)++counts[NODES];
        int s=slots.size();
        if(s==2){
            if(slots[0]<0 || slots[1]<0)return;
            size_t id=key(slots[0],slots[1]);
            if(done[id] || groups[id].empty())return;
            done[id]=1;if constexpr(audit)++counts[LEAVES];
            for(int e:groups[id]){
                out[e]=div(1,add(bg.at(0,1),exclusive[e]));
                if(path_data){
                    int* trace=path_data+4*e;
                    trace[0]=path.degree_cost; trace[1]=path.pivots;
                    trace[2]=path.levels; trace[3]=path.max_degree;
                }
            }
            return;
        }
        const int quarter=s/4;
        for(int u=0;u<4;++u)for(int w=u+1;w<4;++w){
            Ids keep,child;std::vector<char> inside(s);bool needed=false;
            for(int i=0;i<s;++i)if(i/quarter==u || i/quarter==w){
                keep.push_back(i);child.push_back(slots[i]);inside[i]=1;
            }
            // Query presence is tested by pair index, never by rescanning the k edges.
            for(size_t j=0;j<child.size() && !needed;++j)if(child[j]>=0)
                for(size_t i=0;i<j;++i)if(child[i]>=0){
                    size_t id=key(child[i],child[j]);
                    if(!done[id] && !groups[id].empty()){needed=true;break;}
                }
            if(!needed)continue;
            Matrix g=bg;
            for(int j=0;j<s;++j)if(slots[j]>=0)for(int i=0;i<j;++i)
                if(slots[i]>=0 && !(inside[i] && inside[j])){
                    double x=pairs.at(slots[i],slots[j]);
                    if(x>0)g.set(i,j,add(g.at(i,j),x));
                }
            PathStats child_path=path;
            ++child_path.levels;
            Matrix next=reduce(g,slots,keep,path_data ? &child_path : nullptr);
            vertex_visit(child,next,child_path);
        }
    }
    void edge_visit(const Ids& queries,const Ids& slots,const Matrix& bg){
        if constexpr(audit)++counts[NODES];
        if(queries.size()==1){out[queries[0]]=div(1,bg.at(0,1));return;}
        size_t cut=queries.size()/2;
        Ids parts[2]={{queries.begin(),queries.begin()+cut},{queries.begin()+cut,queries.end()}};
        int s=slots.size();Ids local(v,-1);
        for(int i=0;i<s;++i)local[slots[i]]=i;
        for(int side=0;side<2;++side){
            Matrix g=bg;std::vector<Vec> buckets(size_t(s)*s);
            for(int e:parts[1-side]){
                int x=local[a[e]],y=local[b[e]];
                buckets[size_t(std::min(x,y))*s+std::max(x,y)].push_back(weight[e]);
            }
            for(int j=0;j<s;++j)for(int i=0;i<j;++i){
                auto& bucket=buckets[size_t(i)*s+j];
                if(!bucket.empty())g.set(i,j,add(g.at(i,j),sum(bucket)));
            }
            std::vector<char> wanted(s);
            for(int e:parts[side])wanted[local[a[e]]]=wanted[local[b[e]]]=1;
            Ids keep,child;
            for(int i=0;i<s;++i)if(wanted[i]){keep.push_back(i);child.push_back(slots[i]);}
            Matrix next=reduce(g,slots,keep);edge_visit(parts[side],child,next);
        }
    }
    void factor_solve(int edge){
        int ground=b[edge];Ids order;
        for(int i=0;i<v;++i)if(i!=ground)order.push_back(i);
        order.push_back(ground);int source=std::find(order.begin(),order.end(),a[edge])-order.begin();
        Matrix g(v),p(v);
        for(int j=0;j<v;++j)for(int i=0;i<j;++i)g.set(i,j,pairs.at(order[i],order[j]));
        g.set(source,v-1,exclusive[edge]);
        Vec d(v-1),y(v-1),x(v-1);
        for(int h=0;h<v-1;++h){
            Vec row;for(int i=h+1;i<v;++i)if(g.at(h,i)>0)row.push_back(g.at(h,i));
            d[h]=sum(row);
            for(int i=h+1;i<v;++i)if(g.at(h,i)>0)p.at(i,h)=div(g.at(h,i),d[h]);
            for(int i=h+1;i<v;++i)if(p.at(i,h)>0)for(int j=i+1;j<v;++j)
                if(g.at(h,j)>0)g.set(i,j,add(g.at(i,j),mul(p.at(i,h),g.at(h,j))));
        }
        for(int i=0;i<v-1;++i){
            y[i]=(i==source);
            for(int j=0;j<i;++j)if(p.at(i,j)>0 && y[j]>0)y[i]=add(y[i],mul(p.at(i,j),y[j]));
        }
        for(int i=v-2;i>=0;--i){
            x[i]=div(y[i],d[i]);
            for(int j=i+1;j<v-1;++j)if(p.at(j,i)>0 && x[j]>0)x[i]=add(x[i],mul(p.at(j,i),x[j]));
        }
        out[edge]=x[source];
    }
    void run(int mode){
        Ids ids,slots(v);std::iota(slots.begin(),slots.end(),0);
        for(int e=0;e<k;++e){
            if(a[e]<0 || b[e]<0 || a[e]>=v || b[e]>=v || !(r[e]>0) || !std::isfinite(r[e]))throw std::runtime_error("invalid input");
            if(a[e]==b[e]){out[e]=0;continue;}
            weight[e]=div(1,r[e]);ids.push_back(e);groups[key(a[e],b[e])].push_back(e);
        }
        if(ids.empty())return;
        if(mode==2){edge_visit(ids,slots,Matrix(v));return;}
        assemble_pairs();
        if(mode==4){for(int e:ids)factor_solve(e);return;}
        if(mode!=3)throw std::runtime_error("unknown mode");
        int power=2;while(power<v)power*=2;slots.resize(power,-1);
        vertex_visit(slots,Matrix(power));
    }
    Solver(int n,int m,const int* from,const int* to,const double* rr,double* oo,std::uint64_t* cc):
        v(n),k(m),a(from),b(to),r(rr),out(oo),counts(cc),weight(m),exclusive(m),pairs(n),groups(size_t(n)*n),done(size_t(n)*n){}
};
extern "C" int kernel(int mode,int audit,int v,int k,const int* a,const int* b,const double* r,double* out,std::uint64_t* counts){
    std::fill(counts,counts+COUNT,0);
    int previous=std::fetestexcept(FE_ALL_EXCEPT),status=0;
    std::feclearexcept(FE_ALL_EXCEPT);
    try{
        if(audit)Solver<true>(v,k,a,b,r,out,counts).run(mode);
        else Solver<false>(v,k,a,b,r,out,counts).run(mode);
    }catch(...){status=1;}
    counts[HW_UNDERFLOW]=std::fetestexcept(FE_UNDERFLOW)!=0;
    std::feclearexcept(FE_ALL_EXCEPT);if(previous)std::feraiseexcept(previous);
    return status;
}

extern "C" int kernel_paths(int v,int k,const int* a,const int* b,const double* r,double* out,int* paths){
    std::uint64_t counts[COUNT]={};
    std::fill(paths,paths+4*k,0);
    try{
        Solver<false> solver(v,k,a,b,r,out,counts);
        solver.path_data=paths;
        solver.run(3);
        return 0;
    }catch(...){return 1;}
}
