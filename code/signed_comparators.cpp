// Instrumentable binary64 grounded-inverse and sparse rank-one update baselines.
// Inputs are compressed endpoint indices and resistances; output is the deleted
// endpoint resistance. Invalid/disconnected kernels or nonpositive pivots fail.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

using Vector = std::vector<double>;
using Row = std::map<int,double>;

template<bool audit> struct Arithmetic {
    std::uint64_t count=0;
    double record(double x){if constexpr(audit)++count; return x;}
    double add(double x,double y){return record(x+y);}
    double sub(double x,double y){return record(x-y);}
    double mul(double x,double y){return record(x*y);}
    double div(double x,double y){return record(x/y);}
    double root(double x){return record(std::sqrt(x));}
};

template<bool audit> void dense_inverse(int n,int k,const int* a,const int* b,const double* r,
                                        double* out,Arithmetic<audit>& op) {
    int m=n-1;
    Vector L(m*m),Q(n*n),col(m);
    for(int e=0;e<k;++e){
        int i=a[e],j=b[e];
        if(i==j)continue;
        double g=op.div(1,r[e]);
        if(i<m)L[i*m+i]=op.add(L[i*m+i],g);
        if(j<m)L[j*m+j]=op.add(L[j*m+j],g);
        if(i<m && j<m)L[std::max(i,j)*m+std::min(i,j)]=op.sub(L[std::max(i,j)*m+std::min(i,j)],g);
    }
    for(int i=0;i<m;++i)for(int j=0;j<=i;++j){
        double value=L[i*m+j];
        for(int h=0;h<j;++h)value=op.sub(value,op.mul(L[i*m+h],L[j*m+h]));
        if(i==j && !(value>0))throw std::runtime_error("dense pivot");
        L[i*m+j]=i==j?op.root(value):op.div(value,L[j*m+j]);
    }
    for(int q=0;q<m;++q){
        for(int i=0;i<m;++i){
            double t=(i==q);
            for(int j=0;j<i;++j)t=op.sub(t,op.mul(L[i*m+j],col[j]));
            col[i]=op.div(t,L[i*m+i]);
        }
        for(int i=m-1;i>=0;--i){
            double t=col[i];
            for(int j=i+1;j<m;++j)t=op.sub(t,op.mul(L[j*m+i],Q[j*n+q]));
            Q[i*n+q]=op.div(t,L[i*m+i]);
        }
    }
    for(int e=0;e<k;++e){
        int i=a[e],j=b[e];
        if(i==j){out[e]=0;continue;}
        double R=op.sub(op.add(Q[i*n+i],Q[j*n+j]),op.mul(2,Q[i*n+j]));
        out[e]=op.div(R,op.sub(1,op.div(R,r[e])));
    }
}

template<bool audit> void sparse_update(int n,int k,const int* a,const int* b,const double* r,
                                        double* out,Arithmetic<audit>& op) {
    const int m=n-1;
    Vector d(m),x(n);
    std::vector<Row> adj(m);
    for(int e=0;e<k;++e){
        int i=a[e],j=b[e];
        if(i==j)continue;
        const double g=op.div(1,r[e]);
        if(i<m)d[i]=op.add(d[i],g);
        if(j<m)d[j]=op.add(d[j],g);
        if(i<m && j<m)adj[i][j]=adj[j][i]=op.sub(adj[i][j],g);
    }
    std::vector<int> order;
    std::vector<bool> live(m,true);
    std::vector<std::vector<std::pair<int,double>>> factors(m);
    for(int step=0;step<m;++step){
        int p=-1;
        for(int i=0;i<m;++i)if(live[i] && (p<0 || adj[i].size()<adj[p].size()))p=i;
        if(!(d[p]>0))throw std::runtime_error("sparse pivot");
        live[p]=false; order.push_back(p);
        const Row& row=adj[p];
        for(auto j=row.begin();j!=row.end();++j){
            auto [i,v]=*j; double mult=op.div(v,d[p]); factors[p].emplace_back(i,mult);
            d[i]=op.sub(d[i],op.mul(mult,v)); adj[i].erase(p);
            for(auto h=std::next(j);h!=row.end();++h){
                auto [z,w]=*h;
                adj[i][z]=adj[z][i]=op.sub(adj[i][z],op.mul(mult,w));
            }
        }
        adj[p].clear();
    }
    for(int e=0;e<k;++e){
        if(a[e]==b[e]){out[e]=0;continue;}
        std::fill(x.begin(),x.end(),0);
        if(a[e]<m)x[a[e]]=1;
        if(b[e]<m)x[b[e]]=-1;
        for(int p:order)if(x[p]!=0)
            for(auto [i,mult]:factors[p])x[i]=op.sub(x[i],op.mul(mult,x[p]));
        for(int p:order)x[p]=op.div(x[p],d[p]);
        for(auto it=order.rbegin();it!=order.rend();++it){
            int p=*it;
            for(auto [i,mult]:factors[p])if(x[i]!=0)x[p]=op.sub(x[p],op.mul(mult,x[i]));
        }
        double R=op.sub(x[a[e]],x[b[e]]);
        out[e]=op.div(R,op.sub(1,op.div(R,r[e])));
    }
}

template<bool audit> void run(int mode,int n,int k,const int* a,const int* b,const double* r,double* out,
                              std::uint64_t* count){
    Arithmetic<audit> op;
    if(mode==5)dense_inverse(n,k,a,b,r,out,op);
    else if(mode==6)sparse_update(n,k,a,b,r,out,op);
    else throw std::runtime_error("mode");
    *count=op.count;
}

extern "C" int comparison(int mode,int audit,int n,int k,const int* a,const int* b,const double* r,
                          double* out,std::uint64_t* count){
    try{
        if(n<1 || k<0)throw std::runtime_error("kernel size");
        for(int e=0;e<k;++e)if(a[e]<0 || b[e]<0 || a[e]>=n || b[e]>=n || !(r[e]>0) || !std::isfinite(r[e]))
            throw std::runtime_error("data");
        if(audit)run<true>(mode,n,k,a,b,r,out,count);
        else run<false>(mode,n,k,a,b,r,out,count);
        return 0;
    }catch(...){return 1;}
}
