# Experiment registry

`registry.jsonl` is append-only. Each row binds a hypothesis and conclusion to
the Git state, config, immutable seed manifest, dataset/checkpoint hashes,
metrics, and runtime. Failed experiments stay in the registry.

The protected `final_test` split in
`configs/seeds/natural_treechop_v1.json` is rejected by tooling unless a caller
passes the explicit project-owner approval flag. Normal development scripts do
not expose that flag.
