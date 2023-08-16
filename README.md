# bioio-czi

[![Build Status](https://github.com/bioio-devs/bioio-czi/workflows/CI/badge.svg)](https://github.com/bioio-devs/bioio-czi/actions)
[![Documentation](https://github.com/bioio-devs/bioio-czi/workflows/Documentation/badge.svg)](https://bioio-devs.github.io/bioio-czi)

A Bioio reader plugin for reading czi images.

This plugin is intended to be used in conjunction with [bioio](https://github.com/bioio-devs/bioio)
---

## Installation

**Stable Release:** `pip install bioio-czi`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-czi.git`

## Quickstart

```python
from bioio_czi import Reader 

r = Reader("my-image.ext")
r.dims
```

## Documentation

For full package documentation please visit [bioio-devs.github.io/bioio-czi](https://bioio-devs.github.io/bioio-czi).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
