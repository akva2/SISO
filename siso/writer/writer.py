from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from inspect import isabstract
from pathlib import Path

import numpy as np
from singledispatchmethod import singledispatchmethod
import treelog as log

from typing import Any, Optional, Dict, Union, List
from ..typing import Array2D, StepData

from .. import config
from ..geometry import Patch, UnstructuredPatch, GeometryManager
from ..fields import Field, PatchData, FieldData, SimpleField, CombinedField
from ..filters import Filter, StepFilter, FieldFilter
from ..util import subclasses



class Writer(Filter, StepFilter):

    writer_name: str

    _geometry: GeometryManager
    outpath: Path

    stepid: int
    stepdata: Dict[str, Any]
    step_finalized: bool
    geometry_finalized: bool

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        """Return true if the class can handle the given format."""
        return False

    @staticmethod
    def find_applicable(fmt: str) -> type:
        """Return a writer subclass that can handle the given format."""
        for cls in subclasses(Writer, invert=True):
            if isabstract(cls):
                continue
            if cls.applicable(fmt):
                log.info(f"Using writer: {cls.writer_name}")
                return cls
            else:
                log.debug(f"Rejecting writer: {cls.writer_name}")
        raise TypeError(f"Unable to find any applicable writers for {fmt}")

    def __init__(self, outpath: Path):
        self._geometry = GeometryManager()
        self.outpath = Path(outpath)

    def validate(self):
        """Raise an error if config options are invalid."""
        pass

    def make_filename(self, root: Optional[Path] = None, with_step: bool = False, indexing: int = 1):
        """Create a filename based on the output path, including step ID or
        not, if required.  If ROOT is not given, the configured output
        path is used.
        """
        if root is None:
            root = self.outpath
        if not (with_step and config.multiple_timesteps):
            return root
        return root.with_name(f'{root.stem}-{self.stepid + indexing}').with_suffix(root.suffix)

    def __enter__(self):
        self.stepid = -1
        self.stepdata = dict()
        self.step_finalized = True
        self.geometry_finalized = True
        return self

    def __exit__(self, tp, value, bt):
        pass

    @contextmanager
    def step(self, stepdata: StepData):
        assert self.step_finalized
        self.stepid += 1
        self.stepdata = stepdata
        self.step_finalized = False
        self.geometry_finalized = False

        yield self

        assert self.geometry_finalized
        assert not self.step_finalized
        self.step_finalized = True

    @contextmanager
    def geometry(self, field: Field):
        self.geometry_finalized = False
        yield partial(self.update_geometry, field)
        self.geometry_finalized = True

    @contextmanager
    def field(self, field: Field):
        assert self.geometry_finalized
        yield partial(self.update_field, field)

    def update_geometry(self, geometry: Field, patch: Patch, data: Array2D):
        """Call this after add_step to update the geometry for each new patch.
        This method only returns the global patch ID.  It should be
        reimplemented in subclasses.
        """
        assert not self.geometry_finalized
        return self._geometry.update(patch, data)

    @abstractmethod
    def update_field(self, field: Field, patch: PatchData, data: FieldData):
        """Call this method after finalize_geometry to issue updates to fields
        which are defined on patches.
        """
        pass


class TesselatedWriter(Writer):

    @abstractmethod
    def _update_geometry(self, patchid: int, patch: UnstructuredPatch, data: Array2D):
        pass

    def update_geometry(self, geometry: Field, patch: Patch, data: Array2D):
        patchid = super().update_geometry(geometry, patch, data)
        self._update_geometry(patchid, patch.tesselate(), patch.tesselate_field(data))

    @abstractmethod
    def _update_field(self, field: Field, patchid: int, data: Array2D):
        pass

    @singledispatchmethod
    def update_field(self, field: Field, patch: PatchData, data: FieldData):
        raise NotImplementedError

    @update_field.register(SimpleField)
    def _(self, field: SimpleField, patch: Patch, data: Array2D):
        patchid = self._geometry.global_id(patch)
        data = patch.tesselate_field(data, cells=field.cells)
        self._update_field(field, patchid, data)

    @update_field.register(CombinedField)
    def _(self, field: CombinedField, patch: List[Patch], data: List[Array2D]):
        patchid = self._geometry.global_id(patch[0])
        data = np.hstack([p.tesselate_field(d, cells=field.cells) for p, d in zip(patch, data)])
        self._update_field(field, patchid, data)
