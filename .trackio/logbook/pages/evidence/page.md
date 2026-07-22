# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2f8cd0a9bb59", "created_at": "2026-07-22T08:04:06+00:00", "title": "Verification output (last 40 lines)"}
-->
## Verification output (last 40 lines)

```
  Gram ranks across 3 layers: [np.int64(2), np.int64(2), np.int64(2)] (consistent: True)
  -> PASS

==============================================================================
CLAIM 6: low-dim spectra → fast convergence (proxy)
==============================================================================
  loss: first=0.8212, last=0.8212 (converges)
  -> FAIL

==============================================================================
VERDICT SUMMARY
==============================================================================
  [PASS] c1_hessian_rank
  [FAIL] c2_gn_decomposition
  [PASS] c3_gradient_directions
  [PASS] c4_gram_rank
  [PASS] c5_cross_layer
  [FAIL] c6_convergence

  4/6 claims verified.
Traceback (most recent call last):
  File "/home/dineshai/Drives/Code/AllCode/ReproduceICML/papers/icml26-repro-RwiGcN2feP-low-dim-spectra/repro/src/verify_spectra.py", line 104, in <module>
    json.dump(results, open(os.path.join(OUT, "verdict.json"), "w"), indent=2)
  File "/home/dineshai/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/json/__init__.py", line 179, in dump
    for chunk in iterable:
                 ^^^^^^^^
  File "/home/dineshai/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/json/encoder.py", line 432, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/home/dineshai/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/home/dineshai/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/home/dineshai/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/json/encoder.py", line 326, in _iterencode_list
    yield from chunks
  File "/home/dineshai/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/json/encoder.py", line 439, in _iterencode
    o = _default(o)
        ^^^^^^^^^^^
  File "/home/dineshai/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type int64 is not JSON serializable
```
