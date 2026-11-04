# AcrAB-TolC-type test case

A predicted tripartite pump (`cpx_4264`, B-factors are pLDDT): three RND transporters (1037 residues), six adaptors and three TolC. It is the test case for `-dm` on a complex that does not lie along z in its input file (its long axis is 46° from z), which MemPrO oriented with the membranes touching nothing until `-la` turned the protein onto z before the search (`tests/test_dual_membrane.py`).

| File | What it is |
|---|---|
| `cpx_4264-cg.pdb.gz` | Martini 3 coarse-grained model (martinize2 `-merge all -elastic -scfix -dssp`, OXT atoms removed), 6,495 residues. |

The predicted model reuses chain letters: each adaptor is split into its α-hairpin (residues 96–175) and the rest, renumbered from 1, under one chain letter. martinize2 pools the atoms of repeated residue numbers, so before coarse-graining a new chain was started wherever the numbering goes backwards (18 chains). The adaptors' unprocessed signal peptides sit in the inner membrane.

Unzip first (`gunzip -k cpx_4264-cg.pdb.gz`). Orient (about 2 min with `-ng 16` on an M4 Mac):

    mempro -f cpx_4264-cg.pdb -o Orient -ng 16 -ni 150 -dm

The membranes come out about 275 Å apart, TolC's barrel in the membrane at +z and the RND transporters' helices in the one at -z.
