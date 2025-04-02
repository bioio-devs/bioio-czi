from typing import Dict, Tuple

# Example bounding box: {'X': (0, 100), 'Y': (0, 100), 'Z': (0, 20)}
BoundingBox = Dict[str, Tuple[int, int]]


def size(bounding_box: BoundingBox, dim: str) -> int:
    """
    Return the size of the dimension if it is in the bounding box, otherwise -1.
    """
    if dim not in bounding_box:
        return -1
    bounds = bounding_box[dim]
    return bounds[1] - bounds[0]
