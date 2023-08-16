# czi-to-ome-xslt

[![Build Status](https://github.com/AllenCellModeling/czi-to-ome-xslt/workflows/Test%20Main/badge.svg)](https://github.com/AllenCellModeling/czi-to-ome-xslt/actions/test-main.yml)

XSLT files to convert from CZI (Zeiss) microscopy image metadata
to OME image metadata schema.

---

## Usage

If you want to use this work in a standalone fashion we recommend submoduling
this repository into your own repo:

```bash
git submodule add https://github.com/AllenCellModeling/czi-to-ome-xslt.git
```

You can then run the transformation in any language of your choosing.

### Python

_**Requires `lxml`** (`pip install lxml`)_

```python
import lxml.etree as ET

# Parse template and generate transform function
template = ET.parse("czi-to-ome-xslt/xslt/czi-to-ome.xsl")
transformer = ET.XSLT(template)

# Parse CZI XML
czixml = ET.parse("your-czi-metadata.xml")

# Transform
omexml = transformer(czixml)

# Write to file
with open("your-converted-czi-metadata.ome.xml", "w") as open_f:
    open_f.write(str(omexml))
```

This work has already been incorporated into
[`aicsimageio`](https://github.com/AllenCellModeling/aicsimageio).

```python
from aicsimageio import AICSImage

img = AICSImage("your-file.czi")
img.ome_metadata
```

## EXSLT
This work utilizes the EXSLT extensions for XSLT 1.0 (for example, the `str:tokenize` function). Popular XML libraries
such as `lxml` have built in support for EXSLT. To use these extensions, include the necessary attributes on `<xsl:stylesheet>` in the
file you are working on:
```
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:ome="http://www.openmicroscopy.org/Schemas/OME/2016-06"
    xmlns:str="http://exslt.org/strings" extension-element-prefixes="str">
    ...
```
You can then use an EXSLT function like so:
```
<xsl:value-of select="str:tokenize(/Some/Path, ',')" />
```

For more information on using EXSLT and the functions avaialbe, see the [EXSLT docs](http://exslt.org/howto.html).

## Comparison with Bioformats

For full metadata comparison between XSLT and Bioformats, see
[comparison](./docs/comparison).


## Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

Please feel free to contribute documentation, new additions to the XSLT itself,
and/or tests!

**We especially encourage contributing examples of using this work in languages other
than Python**.

For instructions on how to contribute new additions to the XSLT and/or tests, see
[CONTRIBUTING](./docs/CONTRIBUTING.md).


---


***Free software: BSD license***
