from pathlib import Path
import pytest

from click.testing import CliRunner

from .shared import TESTCASES, compare_vtk_unstructured, PreparedTestCase

import vtk


def load_grid(filename: Path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    return reader.GetOutput()


def compare_vtu(out: Path, ref: Path):
    assert out.exists()
    assert ref.exists()
    compare_vtk_unstructured(load_grid(out), load_grid(ref))


@pytest.mark.parametrize('case', TESTCASES['vtu'])
def test_vtu_integrity(case: PreparedTestCase):
    with case.invoke('vtu') as tempdir:
        for out, ref in case.check_files(tempdir):
            compare_vtu(out, ref)