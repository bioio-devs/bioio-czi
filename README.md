# bioio-czi

[![Build Status](https://github.com/bioio-devs/bioio-czi/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-czi/actions)
[![PyPI version](https://badge.fury.io/py/bioio-czi.svg)](https://badge.fury.io/py/bioio-czi)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO reader plugin for reading CZIs using `pylibczirw` (default) or `aicspylibczi`.

---


## Documentation

[See the bioio documentation on our GitHub pages site](https://bioio-devs.github.io/bioio/OVERVIEW.html) - the general use and installation instructions there will work for this package.

Information about the base reader this package relies on can be found in the `bioio-base` repository [here](https://github.com/bioio-devs/bioio-base).

This plugin attempts to follow the latest specification for the CZI file format from
Carl Zeiss Microscopy ([currently v1.2](./docs/2024_06_02_DS_ZISRAW-FileFormat.pdf)).

## Installation

Install bioio-czi alongside bioio:

`pip install bioio bioio-czi`

**Stable Release:** `pip install bioio-czi`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-czi.git`

## pylibczirw vs. aicspylibczi
`bioio-czi` can operate in [pylibczirw](https://github.com/ZEISS/pylibczirw) mode (the default) or [aicspylibczi](https://github.com/AllenCellModeling/aicspylibczi) mode.

| Feature | pylibczirw mode | aicspylibczi mode |
|--|--|--|
| Read CZIs from the internet | ✅ | ❌ |
| Read single tile from tiled CZI | ❌ | ✅ |
| Read single tile's metadata from tiled CZI | ❌ | ✅ |
| Read elapsed time metadata* | ❌ | ✅ |
| Handle CZIs with different dimensions per scene** | ❌ | ✅ |
| Read stitched mosaic of a tiled CZI | ✅ | ✅ |

The primary difference is that `pylibczirw` supports reading CZIs over the internet but cannot access individual tiles from a tiled CZI. To use `aicspylibczi`, add the `use_aicspylibczi=True` parameter when creating a reader. For example: `from bioio import BioImage; img = BioImage(..., use_aicspylibczi=True)`.

*Elapsed time metadata include the following. These are derived from individual subblock metadata.
* `BioImage(...).time_interval`
* `BioImage(...).standard_metadata.timelapse_interval`
* `BioImage(...).standard_metadata.total_time_duration`

**The underlying pylibczirw reader assumes that each scene has the same dimension. Files that do not meet this requirement may be read incorrectly in pylibczirw mode.

## Example Usage (see full documentation for more examples)

### Basic usage
```python
from bioio import BioImage

path = (
    "https://allencell.s3.amazonaws.com/aics/hipsc_12x_overview_image_dataset/"
    "stitchedwelloverviewimagepath/05080558_3500003720_10X_20191220_D3.czi"
)

img = BioImage(path)
print(img.shape)  # (1, 1, 1, 5684, 5925)
```
Note: accessing files from the internet is not available in `aicspylibczi` mode.

### Individual tiles with aicspylibczi
```python
img = BioImage(
    "S=2_4x2_T=2=Z=3_CH=2.czi",
    reconstruct_mosaic=False,
    include_subblock_metadata=True,
    use_aicspylibczi=True
)
print(img.dims)  # <Dimensions [M: 8, T: 2, C: 2, Z: 3, Y: 256, X: 256]>
subblocks = img.metadata.findall("./Subblocks/Subblock")
print(len(subblocks))  # 192
print(img.get_image_data("TCZYX", M=3).shape)  # (2, 2, 3, 256, 256)
```
The `M` dimension is used to select a specific tile.

### Stitched mosaic with pylibczirw
```python
img = BioImage("S=2_4x2_T=2=Z=3_CH=2.czi")
print(img.dims)  # <Dimensions [T: 2, C: 2, Z: 3, Y: 487, X: 947]>
```
All 8 tiles are stitched together. Where tiles overlap, the pixel value is the pixel value from the tile with the highest M-index.

### Explicit Reader
This example shows a simple use case for just accessing the pixel data of the image
by explicitly passing this `Reader` into the `BioImage`. Passing the `Reader` into
the `BioImage` instance is optional as `bioio` will automatically detect installed
plug-ins and auto-select the most recently installed plug-in that supports the file
passed in.
```python
from bioio import BioImage
import bioio_czi

img = BioImage("my_file.czi", reader=bioio_czi.Reader)
img.data
```

## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch) or check this repository's issue tab.


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
