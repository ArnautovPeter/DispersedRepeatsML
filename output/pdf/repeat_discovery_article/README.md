# Concise project report

`article.pdf` is the concise English report requested as a shorter replacement
for the previous research-style article. `article.tex` is the editable source.
It covers the compared methods, actual training procedure, saved ML-versus-k-mer
results, canonical 9-mer/profile discovery, and a brief separate summary of later sequence-based experiments.

No new training, benchmark, or model checks were run for this revision.
The numerical comparison is copied from the existing experiment artifacts.
`data/ml_comparison.tsv`, `data/ml_config.json`, and `data/ml_complete.json`
are included as supporting records. `data/9mer_final_metrics.tsv` and
`data/9mer_validation_metrics.tsv` preserve the separate coordinate-search results.
The report explains token pooling, contrastive projection, and how HDBSCAN
turns normalized embeddings into clusters and noise labels.

Build with `bash build.sh`. Only pdfLaTeX and the packages in article.tex are
required. No data processing or model execution is performed by the build.

The previous full article and its figures remain archived separately in
`../repeat_discovery_full_article_backup/` and
`../repeat_discovery_full_article_bundle.zip`.
