/* Query-global, target-infix affine-gap profile alignment.
 * Natural-log profile scores. Gap cost open+(length-1)*extend.
 * No truth, genomic seed coordinates, or family counts are accepted.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MAX(a,b) ((a)>(b)?(a):(b))
static int fill(const float *w,int m,const unsigned char *s,int n,float go,float ge,
                float *out,unsigned char *trace) {
    float *prev=malloc((n+1)*sizeof(float));
    float *curr=malloc((n+1)*sizeof(float));
    float *del=malloc((n+1)*sizeof(float));
    if(!prev||!curr||!del) {free(prev);free(curr);free(del);return -1;}
    const float neg=-1e30f;
    for(int j=0;j<=n;j++) {prev[j]=0;del[j]=neg;}
    for(int i=1;i<=m;i++) {
        curr[0]=-go-(i-1)*ge;
        float ins=neg;
        for(int j=1;j<=n;j++) {
            float dopen=prev[j]-go,dext=del[j]-ge;
            int de=dext>dopen;
            del[j]=de?dext:dopen;
            float iopen=curr[j-1]-go,iext=ins-ge;
            int ie=iext>iopen;
            ins=ie?iext:iopen;
            float best=prev[j-1]+w[(i-1)*4+s[j-1]];
            unsigned char direction=1;
            if(del[j]>best) {best=del[j];direction=2;}
            if(ins>best) {best=ins;direction=3;}
            curr[j]=best;
            if(trace) trace[(size_t)i*(n+1)+j]=direction|(de<<2)|(ie<<3);
        }
        float *temp=prev;prev=curr;curr=temp;
    }
    memcpy(out,prev+1,n*sizeof(float));
    free(prev);free(curr);free(del);
    return 0;
}
int gapped_profile_scores(const float *w,int m,const unsigned char *s,int n,
                          float go,float ge,float *out) {
    return fill(w,m,s,n,go,ge,out,NULL);
}
int gapped_profile_trace(const float *w,int m,const unsigned char *s,int n,
                         float go,float ge,int *start,float *score,char *ops) {
    unsigned char *trace=calloc((size_t)(m+1)*(n+1),1);
    float *out=malloc(n*sizeof(float));
    if(!trace||!out) {free(trace);free(out);return -1;}
    if(fill(w,m,s,n,go,ge,out,trace)<0) {free(trace);free(out);return -1;}
    *score=out[n-1];
    int i=m,j=n,state=0,k=0;
    while(i>0) {
        if(j==0) {ops[k++]='I';--i;continue;}
        unsigned char bits=trace[(size_t)i*(n+1)+j];
        if(state==0) {
            int direction=bits&3;
            if(direction==1) {ops[k++]='M';--i;--j;}
            else state=direction==2?1:2;
        } else if(state==1) {
            ops[k++]='I';--i;state=(bits&4)?1:0;
        } else {
            ops[k++]='D';--j;state=(bits&8)?2:0;
        }
    }
    *start=j;
    free(trace);free(out);
    return k;
}
