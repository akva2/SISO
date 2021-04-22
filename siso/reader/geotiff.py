from pathlib import Path

import numpy as np

from typing import Iterable, Tuple, Optional
from ..typing import StepData, Array2D

from .reader import Reader
from ..coords import Geodetic
from ..fields import Field, SimpleField, Geometry, FieldPatches
from ..geometry import StructuredTopology, Quad, Patch


class GeoTiffGeometryField(SimpleField):

    name = 'Geometry'
    cells = False

    def __init__(self, data):
        self.data = data
        self.fieldtype = Geometry(Geodetic())

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        tiepoint = self.data.GetGeoTransform()
        heights = self.data.GetRasterBand(1).ReadAsArray()
        xs = np.arange(0, heights.shape[1]) + 0.5
        ys = np.arange(0, heights.shape[0]) + 0.5
        xs, ys = np.meshgrid(xs, ys)
        trf_xs = tiepoint[0] + xs * tiepoint[1] + ys * tiepoint[2]
        trf_ys = tiepoint[3] + xs * tiepoint[4] + ys * tiepoint[5]

        from osgeo.osr import SpatialReference, CoordinateTransformation
        inproj = SpatialReference(self.data.GetProjection())
        outproj = SpatialReference()
        outproj.SetWellKnownGeogCS('WGS84')
        trf = CoordinateTransformation(inproj, outproj)

        out = np.array([trf.TransformPoint(x,y) for x,y in zip(trf_xs.flat, trf_ys.flat)])
        nodes = np.array([out[..., 0], out[..., 1], heights.flatten().astype(float)]).T
        topo = StructuredTopology(tuple(h-1 for h in heights.shape), celltype=Quad())

        yield Patch(('geometry',), topo), nodes


class GeoTiffReader(Reader):

    reader_name = "GeoTIFF"

    filename: Path

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        try:
            import gdal
            assert filename.suffix.lower() in ('.tif', '.tiff')
            gdal.Open(str(filename))
        except:
            return False
        return True

    def __init__(self, filename: Path):
        self.filename = filename

    def __enter__(self):
        import gdal
        self.data = gdal.Open(str(self.filename))
        return self

    def __exit__(self, *args):
        pass

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        yield (0, {'time': 0.0})

    def fields(self) -> Iterable[Field]:
        yield GeoTiffGeometryField(self.data)
