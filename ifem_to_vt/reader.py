import h5py
from collections import namedtuple
from io import StringIO
from itertools import chain, product
import logging
import numpy as np
import splipy.io
from splipy import SplineObject, BSplineBasis
from splipy.SplineModel import ObjectCatalogue
import splipy.utils


class G2Object(splipy.io.G2):

    def __init__(self, fstream, mode):
        self.fstream = fstream
        self.onlywrite = mode == 'w'
        super(G2Object, self).__init__('')

    def __enter__(self):
        return self

POINTVALUES = 0
CELLVALUES = 1
EIGENMODE = 2

Field = namedtuple('Field', ['name', 'basis', 'ncomps', 'kind'])
Basis = namedtuple('Basis', ['name', 'updates'])


class Reader:

    def __init__(self, filename):
        self.filename = filename
        self.patch_cache = {}

    def __enter__(self):
        self.h5 = h5py.File(self.filename, 'r')
        self.check()
        self.catalogue = ObjectCatalogue(self.max_pardim)
        return self

    def __exit__(self, type_, value, backtrace):
        self.h5.close()

    def write(self, w):
        if self.modes:
            for bname, field in self.modes.items():
                self.write_geometry(w, 0, self.bases[bname])
                for mid in self.modeids(bname):
                    logging.debug('Writing mode %d', mid)
                    self.write_mode(w, mid, field)
            return

        for lid, time, lgrp in self.times():
            logging.debug('Level %d', lid)
            for bname, bgrp in lgrp.items():
                basis = self.bases[bname]
                if 'basis' in bgrp:
                    logging.debug('Update found for basis %s', basis.name)
                    self.write_geometry(w, lid, basis)
                w.add_time(time)
                for fname in chain(bgrp.get('fields', []), bgrp.get('knotspan', [])):
                    field = self.fields[fname]
                    logging.debug('Updating field %s', field.name)
                    self.write_field(w, lid, field)
                if bname in self.modes:
                    mgrp = bgrp['Eigenmode']
                    for mid in range(len(mgrp)):
                        logging.debug('Writing mode %d', mid)
                        self.write_mode(w, mid, self.modes[bname])

    def _tesselated_patch(self, lid, basis, pid):
        patch = self.patch(lid, basis, pid).clone()
        nodeview = self.catalogue.lookup(patch)
        orig_tesselation = list(nodeview.node.tesselation)
        for i, f in enumerate(nodeview.orientation.flip):
            if f:
                orig_tesselation[i] = orig_tesselation[i][::-1]
        tesselation = [
            orig_tesselation[nodeview.orientation.perm_inv[q]]
            for q in range(len(orig_tesselation))
        ]
        return patch, tesselation

    def _tesselate(self, patch, tesselation, coeffs, cells=False, vectorize=False):
        if cells:
            # Make a piecewise constant patch
            bases = [BSplineBasis(1, kts) for kts in patch.knots()]
            shape = tuple(b.num_functions() for b in bases)
            coeffs = splipy.utils.reshape(coeffs, shape, order='F')
            patch = SplineObject(bases, coeffs, False, raw=True)
            tesselation = [
                [(a+b)/2 for a, b in zip(t[:-1], t[1:])]
                for t in tesselation
            ]
        else:
            coeffs = splipy.utils.reshape(coeffs, patch.shape, order='F')
            patch = SplineObject(patch.bases, coeffs, patch.rational, raw=True)

        if patch.dimension == 1 and vectorize:
            patch.set_dimension(3)
            patch.controlpoints[...,-1] = patch.controlpoints[...,0].copy()
            patch.controlpoints[...,0] = 0.0
        elif patch.dimension > 1:
            patch.set_dimension(3)

        return patch(*tesselation)

    def write_mode(self, w, mid, field):
        for pid in range(self.npatches(0, field.basis)):
            patch, tesselation = self._tesselated_patch(0, field.basis, pid)
            coeffs, data = self.mode_coeffs(field, mid, pid)
            raw = self._tesselate(patch, tesselation, coeffs, vectorize=True)
            results = np.ndarray.flatten(raw)
            w.update_mode(results, field.name, pid, **data)

    def write_field(self, w, lid, field):
        for pid in range(self.npatches(lid, field.basis)):
            patch, tesselation = self._tesselated_patch(lid, field.basis, pid)
            coeffs = self.field_coeffs(field, lid, pid)

            raw = self._tesselate(patch, tesselation, coeffs, cells=field.kind==CELLVALUES)
            kind = 'vector' if raw.shape[-1] > 1 else 'scalar'
            results = np.ndarray.flatten(raw)

            w.update_field(results, field.name, pid, kind, cells=field.kind==CELLVALUES)
            if field.ncomps > 1:
                for i in range(field.ncomps):
                    results = np.ndarray.flatten(raw[...,i])
                    w.update_field(results, '{}[{}]'.format(field.name, i+1), pid, kind=kind)

    def write_geometry(self, w, lid, basis):
        for pid in range(self.npatches(lid, basis)):
            patch = self.patch(lid, basis, pid)
            node = self.catalogue.add(patch).node
            if not hasattr(node, 'patchid'):
                node.patchid = None
            if not hasattr(node, 'last_written'):
                node.last_written = -1

            if node.last_written >= lid:
                logging.debug('Skipping patch %d', pid)
                continue
            node.last_written = lid

            patch = node.obj
            node.tesselation = patch.knots()
            nodes = patch(*node.tesselation)

            # Elements
            ranges = [range(k-1) for k in nodes.shape[:-1]]
            nidxs = [np.array(q) for q in zip(*product(*ranges))]
            eidxs = np.zeros((len(nidxs[0]), 2**len(nidxs)))
            if len(nidxs) == 1:
                eidxs[:,0] = nidxs[0]
                eidxs[:,1] = nidxs[0] + 1
            elif len(nidxs) == 2:
                i, j = nidxs
                eidxs[:,0] = np.ravel_multi_index((i, j), nodes.shape[:-1])
                eidxs[:,1] = np.ravel_multi_index((i+1, j), nodes.shape[:-1])
                eidxs[:,2] = np.ravel_multi_index((i+1, j+1), nodes.shape[:-1])
                eidxs[:,3] = np.ravel_multi_index((i, j+1), nodes.shape[:-1])
            elif len(nidxs) == 3:
                i, j, k = nidxs
                eidxs[:,0] = np.ravel_multi_index((i, j, k), nodes.shape[:-1])
                eidxs[:,1] = np.ravel_multi_index((i+1, j, k), nodes.shape[:-1])
                eidxs[:,2] = np.ravel_multi_index((i+1, j+1, k), nodes.shape[:-1])
                eidxs[:,3] = np.ravel_multi_index((i, j+1, k), nodes.shape[:-1])
                eidxs[:,4] = np.ravel_multi_index((i, j, k+1), nodes.shape[:-1])
                eidxs[:,5] = np.ravel_multi_index((i+1, j, k+1), nodes.shape[:-1])
                eidxs[:,6] = np.ravel_multi_index((i+1, j+1, k+1), nodes.shape[:-1])
                eidxs[:,7] = np.ravel_multi_index((i, j+1, k+1), nodes.shape[:-1])

            logging.debug('Writing patch %d', pid)
            node.patchid = w.update_geometry(
                np.ndarray.flatten(nodes), np.ndarray.flatten(eidxs), len(nidxs), node.patchid
            )

    @property
    def ntimes(self):
        return len(self.h5)

    def times(self):
        for level in range(self.ntimes):
            # FIXME: Grab actual time here as second element
            yield level, float(level), self.h5[str(level)]

    def modeids(self, basis):
        yield from range(len(self.h5['0'][basis]['Eigenmode']))

    def basis_level(self, level, basis):
        if not isinstance(basis, Basis):
            basis = self.bases[basis]
        try:
            return next(l for l in basis.updates[::-1] if l <= level)
        except StopIteration:
            raise ValueError('Geometry for basis {} unavailable at timestep {}'.format(basis, index))

    def npatches(self, level, basis):
        if not isinstance(basis, Basis):
            basis = self.bases[basis]
        level = self.basis_level(level, basis)
        return len(self.h5['{}/{}/basis'.format(str(level), basis.name)])

    def patch(self, lid, basis, index):
        if not isinstance(basis, Basis):
            basis = self.bases[basis]
        lid = self.basis_level(lid, basis)
        key = (lid, basis.name, index)
        if key not in self.patch_cache:
            g2str = self.h5[
                '{}/{}/basis/{}'.format(str(lid), basis.name, str(index+1))
            ][:].tobytes().decode()
            g2data = StringIO(g2str)
            with G2Object(g2data, 'r') as g:
                patch = g.read()[0]
                patch.set_dimension(3)
                self.patch_cache[key] = patch
        return self.patch_cache[key]

    def mode_coeffs(self, field, mid, pid):
        mgrp = self.h5['0'][field.basis.name]['Eigenmode'][str(mid+1)]
        coeffs = mgrp[str(pid+1)][:]
        if 'Value' in mgrp:
            return coeffs, {'value': mgrp['Value'][0]}
        return coeffs, {'frequency': mgrp['Frequency'][0]}

    def field_coeffs(self, field, lid, pid):
        sub = 'fields' if field.kind == POINTVALUES else 'knotspan'
        return self.h5[str(lid)][field.basis.name][sub][field.name][str(pid+1)][:]

    def check(self):
        self.bases = {}
        self.max_pardim = 0
        for lid, _, lgrp in self.times():
            for basis, bgrp in lgrp.items():
                self.bases.setdefault(basis, Basis(basis, []))
                if 'basis' in bgrp:
                    self.bases[basis].updates.append(lid)
                self.max_pardim = max(self.max_pardim, self.patch(lid, basis, 0).pardim)

        self.fields = {}
        self.modes = {}
        basis_iter = ((lid, basis, bgrp) for basis, bgrp in lgrp.items() for lid, _, lgrp in self.times())
        for lid, basis, bgrp in basis_iter:
            if 'fields' in bgrp:
                for field, fgrp in bgrp['fields'].items():
                    if field in self.fields:
                        continue
                    ncomps = len(fgrp['1']) // len(self.patch(lid, basis, 0))
                    self.fields.setdefault(field, Field(field, self.bases[basis], ncomps, POINTVALUES))
            if 'knotspan' in bgrp:
                for field, kgrp in bgrp['knotspan'].items():
                    if field in self.fields:
                        continue
                    patch = self.patch(lid, basis, 0)
                    ncomps = len(kgrp['1']) // np.prod([len(k)-1 for k in patch.knots()])
                    self.fields.setdefault(field, Field(field, self.bases[basis], ncomps, CELLVALUES))
            if 'Eigenmode' in bgrp:
                patch = self.patch(lid, basis, 0)
                self.modes[basis] = Field('Mode Shape', self.bases[basis], patch.dimension, EIGENMODE)
