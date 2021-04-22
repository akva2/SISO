"""Module for G2 format writer."""

import numpy as np
from numpy.linalg import norm
import treelog as log
from splipy.io import G2
from splipy import BSplineBasis, Curve, Surface, Volume

from .. import config
from ..fields import Field
from ..geometry import Patch, StructuredTopology
from ..util import structured_cells
from .writer import Writer

from ..typing import Array, Array2D


class G2Writer(Writer):
    """Simra format writer."""

    writer_name = "G2"

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return fmt == 'g2'

    def validate(self):
        config.require(multiple_timesteps=False)

    def update_geometry(self, geometry: Field, patch: Patch, data: Array2D):
        if not isinstance(patch.topology, StructuredTopology):
            raise TypeError("G2 writer does not support unstructured grids")
        nodeshape = tuple(s+1 for s in patch.topology.shape)
        data = data.reshape(*nodeshape, 3)

        bases = [
            BSplineBasis(2, [0.0] + list(np.arange(n)) + [float(n-1)])
            for n in nodeshape
        ]

        if len(nodeshape) == 1:
            obj = Curve(*bases, data, raw=True)
        elif len(nodeshape) == 2:
            obj = Surface(*bases, data, raw=True)
        elif len(nodeshape) == 3:
            obj = Volume(*bases, data, raw=True)
        else:
            raise TypeError("G2 writer does not support higher-dimensional structures")

        with G2(str(self.outpath)) as f:
            f.write(obj)
        log.user(self.outpath)

    def update_field(self, field: Field, patch: Patch, data: Array2D):
        log.warning("SIMRA writer ignores fields")
