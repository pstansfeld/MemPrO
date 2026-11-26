import os
import tempfile
import unittest

import numpy as np

from mempro.MemPrO import PDB_helper, detect_symmetry_axis


class TestSymmetryAxis(unittest.TestCase):
    def write_cyclic_assembly(self, order, distort=False):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(40, 3))*5
        path = tempfile.NamedTemporaryFile(mode="w",suffix=".pdb",delete=False)
        serial = 0
        for chain_index in range(order):
            angle = chain_index*2*np.pi/order
            rot = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            chain_points = points@rot.T
            if distort and chain_index:
                chain_points += rng.normal(scale=5,size=chain_points.shape)
            for resid,point in enumerate(chain_points,1):
                serial += 1
                path.write("ATOM  %5d  CA  ALA %1s%4d    %8.3f%8.3f%8.3f  1.00 20.00           C\n"
                    %(serial,chr(ord("A")+chain_index),resid,*point))
        path.close()
        return path.name

    def test_detects_cyclic_axis_from_matching_chains(self):
        path = self.write_cyclic_assembly(4)
        try:
            axis,order,rmsd = detect_symmetry_axis(path)
        finally:
            os.unlink(path)
        self.assertEqual(order,4)
        self.assertAlmostEqual(abs(axis[2]),1.0,places=6)
        self.assertLess(rmsd,1e-5)

    def test_rejects_chains_without_cyclic_symmetry(self):
        path = self.write_cyclic_assembly(3,distort=True)
        try:
            with self.assertRaisesRegex(ValueError,"no sufficiently similar|do not support"):
                detect_symmetry_axis(path)
        finally:
            os.unlink(path)

    def test_skips_matching_chains_without_symmetry_to_find_valid_assembly(self):
        rng = np.random.default_rng(42)
        path = tempfile.NamedTemporaryFile(mode="w",suffix=".pdb",delete=False)
        serial = 0
        unrelated = rng.normal(size=(40,3))*5
        cyclic = rng.normal(size=(40,3))*12
        assemblies = [("A",unrelated),("B",unrelated)]
        for chain_index in range(3):
            angle = chain_index*2*np.pi/3
            rot = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            assemblies.append((chr(ord("C")+chain_index),cyclic@rot.T))
        for chain,points in assemblies:
            for resid,point in enumerate(points,1):
                serial += 1
                path.write("ATOM  %5d  CA  ALA %1s%4d    %8.3f%8.3f%8.3f  1.00 20.00           C\n"
                    %(serial,chain,resid,*point))
        path.close()
        try:
            axis,order,_ = detect_symmetry_axis(path.name)
        finally:
            os.unlink(path.name)
        self.assertEqual(order,3)
        self.assertAlmostEqual(abs(axis[2]),1.0,places=6)

    def test_alignment_maps_axis_to_membrane_normal_and_preserves_distances(self):
        helper = PDB_helper("", False, 0, np.empty((0, 2)), False, False)
        helper.surface_poses = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        helper.poses = helper.surface_poses.copy()
        helper.all_poses = helper.surface_poses.copy()
        helper.spheres = helper.surface_poses.copy()
        helper.normals = np.empty((0, 3))
        before = np.linalg.norm(helper.poses[:, None] - helper.poses[None, :], axis=2)

        angle = helper.align_symmetry_axis([1.0, 0.0, 0.0])

        self.assertAlmostEqual(angle, 90.0)
        np.testing.assert_allclose(helper.surface_poses[0], [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(helper.poses[:, None] - helper.poses[None, :], axis=2), before,
        )


if __name__ == "__main__":
    unittest.main()
