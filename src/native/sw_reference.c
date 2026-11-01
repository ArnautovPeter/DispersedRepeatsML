/* Exact scalar affine-gap Smith-Waterman diagnostic backend.
 * Gap cost = open + (length-1)*extend, matching Parasail conventions.
 * out = score, query_begin, query_end, ref_begin, ref_end (half-open).
 * operations is a reverse traceback, one M/I/D per alignment column. */
#include <stdlib.h>
#include <stdint.h>
#define MAX(a,b) ((a)>(b)?(a):(b))
int sw_reference(const char *q, int n, const char *r, int m,
                 int *out, char *operations) {
    size_t cells = (size_t)(n+1)*(m+1);
    int *h = calloc(cells, sizeof(int));
    int *e = calloc(cells, sizeof(int));
    int *f = calloc(cells, sizeof(int));
    if (!h || !e || !f) { free(h); free(e); free(f); return -1; }
    int best=0, bi=0, bj=0, w=m+1;
    for (int i=1; i<=n; ++i) {
        for (int j=1; j<=m; ++j) {
            size_t p=(size_t)i*w+j;
            e[p]=MAX(h[p-w]-5,e[p-w]-2);
            f[p]=MAX(h[p-1]-5,f[p-1]-2);
            int v=MAX(0,h[p-w-1]+(q[i-1]==r[j-1]?2:-1));
            v=MAX(v,MAX(e[p],f[p]));
            h[p]=v;
            /* Earliest ref end, then query end: deterministic tie policy. */
            if (v>best || (v==best && v>0 && (j<bj || (j==bj && i<bi)))) {
                best=v; bi=i; bj=j;
            }
        }
    }
    int i=bi,j=bj,state=0,k=0;
    while(i>0 && j>0) {
        size_t p=(size_t)i*w+j;
        if(state==0) {
            if(h[p]<=0) break;
            if(h[p]==h[p-w-1]+(q[i-1]==r[j-1]?2:-1)) {
                operations[k++]='M'; --i; --j;
            } else if(h[p]==e[p]) state=1;
            else state=2;
        } else if(state==1) {
            int extend = e[p] != h[p-w]-5;
            operations[k++]='I'; --i;
            state=extend?1:0;
        } else {
            int extend = f[p] != h[p-1]-5;
            operations[k++]='D'; --j;
            state=extend?2:0;
        }
    }
    out[0]=best; out[1]=i; out[2]=bi; out[3]=j; out[4]=bj;
    free(h); free(e); free(f);
    return k;
}
