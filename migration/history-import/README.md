# FutureLab V0.48 history import staging

This branch stages the original Git bundle so the repository can restore the complete historical commit graph without rewriting `main`.

- Original bundle SHA-256: `7c57553b2d92908be315a5659ee15658980e628f9693e82bc2e1877be7400284`
- Original bundle size: 32,760,110 bytes
- Bundle fragments: 63 independently Base64-encoded binary segments
- Expected source head: `64b7006f1322ef18b911cf2dba898c5e2e6b1b84`
- Expected source tree: `7e60602b0ff44d4d1abb205fe30f32accc73959a`
- Expected history: 15 commits, 158 tracked files, 6 version tags

The workflow decodes each fragment separately (the fragments have independent Base64 padding), verifies the reconstructed bundle SHA-256 and size, then runs the checked importer. It creates only `import/futurelab-v0.48` and the six version tags. It refuses to overwrite differing refs and verifies that remote `main` remains unchanged.

Do not merge this staging branch into `main`; it is an import runner and payload archive, not a project-source branch.
