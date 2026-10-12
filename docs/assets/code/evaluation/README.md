# LIBERO-Dynamic evaluation code

`libero_dynamic_eval_supplement.zip` is the self-contained evaluation release.
It contains the dynamic evaluation runtime, execution-protocol controllers,
model adapters, paper experiment registry, audit scripts, representative task
gallery, and integrity metadata.

The archive was validated after extraction with its bundled package validator
and six controller/runtime regression tests. Its SHA-256 digest is:

```text
844073ee562432e367e97a957664a445ec3101f8beececdf72f71431a061341e
```

Start with `README.md` inside the archive. The sibling
`../data_generation/` directory contains the uncompressed data-generation
release; the evaluation archive intentionally remains zipped so that its own
internal paths and checksum manifest stay intact.
