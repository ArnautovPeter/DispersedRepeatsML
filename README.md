# DispersedRepeatsML

A research project on discovering repeated DNA sequences and grouping them into families, comparing Transformer embeddings with classical sequence-based methods.

**[Read the report (PDF)](output/pdf/repeat_discovery_article/article.pdf)**

## Methods

- **Nucleotide Transformer:** pretrained DNA embeddings, supervised contrastive learning with a 512 → 256 → 128 projection head, and selective fine-tuning of the final two Transformer blocks.
- **HDBSCAN:** density-based clustering of normalized neural embeddings or canonical 4/5-mer frequency vectors into candidate repeat families, with outliers labelled as noise.
- **9-mer/profile search:** exact sequence seeds followed by iterative statistical profiles for genomic coordinate discovery, with an affine-gap extension for insertions and deletions.

## Results

On 240 held-out human chr22 repeat copies from 11 family labels:

| Representation + HDBSCAN | Pair precision | Pair recall | Pair F1 | ARI |
| --- | ---: | ---: | ---: | ---: |
| Canonical 4/5-mer frequencies | 0.589 | 0.616 | **0.602** | **0.563** |
| Frozen Nucleotide Transformer | 0.225 | 0.502 | 0.311 | 0.215 |
| NT + contrastively trained projection | 0.169 | 0.556 | 0.260 | 0.144 |

Validation selected the projection-stage checkpoint. Neural adaptation did not outperform the k-mer baseline in this experiment.

In a separate 500-kb synthetic genome benchmark, the **9-mer/profile method recovered all 150 copies** through mutation parameter x = 1.5, with locus precision 0.962–1.000 at IoU ≥ 0.5. It returned no loci at x = 2. These coordinate-search results measure a different task from the clustering table above.

With 1% deletions at x = 1.5, gap-aware refinement recovered **149/150 copies**, versus 140/150 before refinement, without false loci. On a real chr22 region, annotation-row recall rose from **37.3% to 41.3%**; this measures agreement with RepeatMasker annotations, not gold-standard precision.

The [report](output/pdf/repeat_discovery_article/article.pdf) describes the training, evaluation protocols and limitations. Its [LaTeX source](output/pdf/repeat_discovery_article/article.tex) and [saved result tables](output/pdf/repeat_discovery_article/data/) are included.
