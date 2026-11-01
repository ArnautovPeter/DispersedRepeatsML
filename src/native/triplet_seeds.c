/* Multi-copy ungapped seed extension, no truth or copy count supplied.
 * Each row is a canonical-oriented context around the same exact 9-mer.
 * The shared 9-mer is excluded from score: candidate selection conditions on it.
 */
#include <math.h>
#include <stdint.h>
static double column(unsigned char a, unsigned char b, unsigned char c,
                     const double *scores) {
    if(a==b && b==c) return scores[0];
    if(a==b || a==c || b==c) return scores[1];
    return scores[2];
}
double best_triplet(const unsigned char *seq, const int *positions, int n,
                    int radius, int k, const double *scores, int *out) {
    int width=2*radius+k;
    double best=-1e30;
    for(int a=0;a<n;a++) for(int b=a+1;b<n;b++) {
        if(positions[b]-positions[a]<width) continue;
        for(int c=b+1;c<n;c++) {
            if(positions[c]-positions[b]<width) continue;
            const unsigned char *s=seq+a*width,*t=seq+b*width,*u=seq+c*width;
            double l=0,r=0,ml=0,mr=0; int il=0,ir=0;
            for(int d=1;d<=radius;d++) {
                l+=column(s[radius-d],t[radius-d],u[radius-d],scores);
                if(l>ml) {ml=l;il=d;}
                int j=radius+k+d-1;
                r+=column(s[j],t[j],u[j],scores);
                if(r>mr) {mr=r;ir=d;}
            }
            if(ml+mr>best) {
                best=ml+mr;
                out[0]=a;out[1]=b;out[2]=c;
                out[3]=radius-il;out[4]=radius+k+ir;
            }
        }
    }
    return best;
}
