# MacAB-TolC test case

MacAB-TolC (PDB 5NIK) spans the cell envelope: TolC's β-barrel in the outer membrane, the MacA hexamer across the periplasm and the MacB dimer in the inner membrane, with its nucleotide-binding domains in the cytoplasm. It is the test case for double-membrane orientation (`-dm`, `tests/test_dual_membrane.py`) and the protein of Insane4MemPrO tutorial 3.

| File | What it is |
|---|---|
| `5NIK-cg.pdb.gz` | Martini 3 coarse-grained model (martinize2 `-merge all -elastic -scfix -dssp`, OXT atoms removed), 4,582 residues. martinize2 writes the chains in a different order from the PDB file, so match residues by position, not by order. |

Unzip first (`gunzip -k 5NIK-cg.pdb.gz`). Orient (about 1 min with `-ng 4`, 3 min with `-ng 16`, on an M4 Mac):

    mempro -f 5NIK-cg.pdb -o Orient -ng 16 -ni 150 -dm

The membranes come out about 272 Å apart, TolC in the membrane at +z (its barrel's extracellular end at about +172 Å) and MacB in the one at -z, with its nucleotide-binding domains below it: the cytoplasm is at -z, as Insane4MemPrO's `-l` (cytoplasmic) and `-uo` (outer membrane, outer leaflet) expect.
