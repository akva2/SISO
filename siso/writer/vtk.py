"""Module for VTK format writers."""

from abc import abstractmethod, abstractclassmethod
from contextlib import contextmanager
from os import makedirs
from pathlib import Path

from typing import TextIO, Optional

import numpy as np
import treelog as log

# We import from vtkmodules to help linters find the module members
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkDataSet, vtkUnstructuredGrid, vtkStructuredGrid, vtkCellArray,
    VTK_HEXAHEDRON, VTK_QUAD, VTK_LINE
)
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridWriter, vtkStructuredGridWriter
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter, vtkXMLStructuredGridWriter
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from .. import config
from ..fields import Field
from ..geometry import StructuredTopology, Hex, Quad, Line, Patch
from ..util import ensure_ncomps, prod
from .writer import Writer

from ..typing import Array2D, StepData



def transpose(data, grid, cells=False):
    if isinstance(grid, vtkStructuredGrid):
        shape = grid.GetDimensions()
        if cells:
            shape = tuple(max(s-1,1) for s in shape)
        data = data.reshape(*shape, -1).transpose(2, 1, 0, 3).reshape(prod(shape), -1)
    return data



class AbstractVTKWriter(Writer):
    """Superclass for all VTK format writers."""

    grid: Optional[vtkDataSet]

    allow_structured: bool
    require_structured: bool

    @staticmethod
    def nan_filter(data: Array2D) -> Array2D:
        """Filter out nans in the data array, if necessary."""
        i, j = np.where(np.isnan(data))
        if len(i) > 0 and config.output_mode == 'ascii':
            log.warning("VTK ASCII files do not support NaN, will be set to zero")
            data[i, j] = 0.0
        return data

    @abstractmethod
    def get_writer(self):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = None

    def update_geometry(self, geometry: Field, patch: Patch, data: Array2D):
        super().update_geometry(geometry, patch, data)

        if not isinstance(patch.topology, StructuredTopology) and self.require_structured:
            raise TypeError(f"{self.writer_name} does not support unstructured grids")

        if isinstance(patch.topology, StructuredTopology) and self.allow_structured:
            if not isinstance(self.grid, vtkStructuredGrid):
                self.grid = vtkStructuredGrid()
            shape = patch.topology.shape
            while len(shape) < 3:
                shape = (*shape, 0)
            if not config.fix_orientation:
                shape = shape[::-1]
            self.grid.SetDimensions(*(s + 1 for s in shape))
        elif not self.grid:
            self.grid = vtkUnstructuredGrid()

        data = ensure_ncomps(self.nan_filter(data), 3, allow_scalar=False)

        if config.fix_orientation:
            data = transpose(data, self.grid)

        points = vtkPoints()
        points.SetData(numpy_to_vtk(data))
        self.grid.SetPoints(points)

        if isinstance(self.grid, vtkUnstructuredGrid):
            if patch.topology.celltype not in [Line(), Quad(), Hex()]:
                raise TypeError(f"Unexpected cell type found: needed line, quad or hex")
            cells = patch.topology.cells
            cells = np.hstack([cells.shape[-1] * np.ones((len(cells), 1), dtype=int), cells]).ravel()
            cells = cells.astype('i8')
            cellarray = vtkCellArray()
            cellarray.SetCells(len(cells), numpy_to_vtkIdTypeArray(cells))
            if patch.topology.celltype == Hex():
                celltype = VTK_HEXAHEDRON
            elif patch.topology.celltype == Quad():
                celltype = VTK_QUAD
            else:
                celltype = VTK_LINE
            self.grid.SetCells(celltype, cellarray)

    def update_field(self, field: Field, patch: Patch, data: Array2D):
        target = self.grid.GetCellData() if field.cells else self.grid.GetPointData()
        data = ensure_ncomps(self.nan_filter(data), 3, allow_scalar=field.is_scalar)
        data = transpose(data, self.grid, cells=field.cells)
        array = numpy_to_vtk(data)
        array.SetName(field.name)
        target.AddArray(array)

    @contextmanager
    def step(self, stepdata: StepData):
        with super().step(stepdata) as step:
            yield step

        filename = self.make_filename(with_step=True)
        writer = self.get_writer()
        writer.SetFileName(str(filename))
        writer.SetInputData(self.grid)
        writer.Write()

        log.user(filename)


class VTKLegacyWriter(AbstractVTKWriter):
    """Writer for VTK legacy format."""

    writer_name = "VTK-legacy"

    allow_structured = True
    require_structured = False

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return fmt == 'vtk'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.require_unstructured:
            self.allow_structured = False

    def validate(self):
        config.require_in(reason="not supported by VTK", output_mode=('binary', 'ascii'))

    def get_writer(self):
        if isinstance(self.grid, vtkStructuredGrid):
            writer = vtkStructuredGridWriter()
        else:
            writer = vtkUnstructuredGridWriter()
        if config.output_mode == 'ascii':
            writer.SetFileTypeToASCII()
        else:
            writer.SetFileTypeToBinary()
        return writer


class VTKXMLWriter(AbstractVTKWriter):
    """Writer for VTK XML-based format."""

    @abstractclassmethod
    def applicable(cls, fmt: str) -> bool:
        pass

    def validate(self):
        super().validate()
        config.require_in(reason=f"not supported by {self.writer_name}", output_mode=('binary', 'ascii', 'appended'))

    @staticmethod
    def nan_filter(data: Array2D) -> Array2D:
        return data

    def get_writer(self):
        if isinstance(self.grid, vtkStructuredGrid):
            writer = vtkXMLStructuredGridWriter()
        else:
            writer = vtkXMLUnstructuredGridWriter()
        if config.output_mode == 'appended':
            writer.SetDataModeToAppended()
        elif config.output_mode == 'ascii':
            writer.SetDataModeToAscii()
        elif config.output_mode == 'binary':
            writer.SetDataModeToBinary()
        return writer


class VTUWriter(VTKXMLWriter):
    """Writer for VTU format (XML-based unstructured grid)."""

    writer_name = "VTU"
    allow_structured = False
    require_structured = False

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return fmt == 'vtu'


class VTSWriter(VTKXMLWriter):
    """Writer for VTS format (XML-based unstructured grid)."""

    writer_name = "VTS"
    allow_structured = True
    require_structured = True

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return fmt == 'vts'


class PVDWriter(VTUWriter):
    """Writer for PVD format (XML-based file with links to other files per timestep)."""

    writer_name = "PVD"

    pvd: TextIO

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return fmt == 'pvd'

    def __init__(self, outpath: Path):
        self.rootfile = outpath
        super().__init__(outpath.with_suffix(f'{outpath.suffix}-data') / 'data.vtu')

    def __enter__(self):
        super().__enter__()
        self.pvd = open(self.rootfile, 'w')
        self.pvd.write('<VTKFile type="Collection">\n')
        self.pvd.write('  <Collection>\n')
        return self

    def __exit__(self, type_, value, backtrace):
        super().__exit__(type_, value, backtrace)
        if value is not None:
            self.pvd.close()
        else:
            self.pvd.write('  </Collection>\n')
            self.pvd.write('</VTKFile>\n')
            self.pvd.close()
            log.user(self.rootfile)

    def make_filename(self, *args, **kwargs):
        filename = super().make_filename(*args, **kwargs)
        makedirs(filename.parent, mode=0o775, exist_ok=True)
        return filename

    @contextmanager
    def step(self, stepdata: StepData):
        with super().step(stepdata) as step:
            yield step
        filename = self.make_filename(with_step=True)
        relative_filename = filename.relative_to(self.rootfile.parent)
        if self.stepdata:
            timestep = next(iter(self.stepdata.values()))
        else:
            timestep = self.stepid
        self.pvd.write('    <DataSet timestep="{}" part="0" file="{}" />\n'.format(timestep, relative_filename))
