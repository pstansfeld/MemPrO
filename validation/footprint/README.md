# MD validation of `-footprint map`

Step 2 of the validation plan in [`docs/footprint-design.md`](../../docs/footprint-design.md). The aim is to check, by simulation, whether `-footprint map` builds membranes whose leaflets are better matched than `-footprint clash` does.

For each protein, both methods build a system with the same box and lipids. Each system is energy-minimised, equilibrated for 0.5 ns NPT with the backbone restrained, and run for a few ns of semi-isotropic NPT (Martini 3, 310 K). The settings follow the group's `cgmd.mdp`.

The comparison does not depend on how the footprint is defined. It counts each leaflet's lipids more than 3 nm from the protein, per nm² of that region. Lipids don't flip between leaflets in Martini, so if one leaflet was built more crowded than the other, it stays denser away from the protein. **Target: the leaflets agree to within 1–2%.**

## Requirements

- The `MemPrO` env, with `mempro` installed (for Insane4MemPrO and `mempro-leaflets`).
- `martinize2` (vermouth ≥ 0.13; e.g. the `CCD2MD` env), set with `--martinize` or `$MARTINIZE2`. Use the same vermouth for both methods and note its version. Since 0.15.0, upstream has made `-el` a hard lower cutoff, which drops elastic bonds shorter than 0.5 nm (about 8% of QmcA's).
- GROMACS (2023 or later), ideally a CUDA build, set with `--gmx` or `$GMX`.
- The Martini 3 force-field files in one directory, set with `--ff` or `$MARTINI3_FF`: `martini_v3.0.0.itp`, `martini_v3.0.0_ions_v1.itp`, `martini_v3.0.0_solvents_v1.itp`, `martini_v3.0.0_phospholipids_v1.itp`, and `martini_v3.0.0_ffbonded_v2.itp` if the phospholipid file uses its named bond parameters (the lipidome version does).

## Running

The examples below are run from this directory. On the SRG machines:

```
export MARTINIZE2=/storage/chem/lfsmgr/SRG/miniforge3/envs/CCD2MD/bin/martinize2
export GMX=/storage/chem/lfsmgr/SRG/mambaforge/envs/gmx2025.4/bin/gmx
export MARTINI3_FF=/path/to/martini3/itps
PY=/storage/chem/lfsmgr/SRG/miniforge3/envs/MemPrO/bin/python
```

### 4G1U (HmuUV, ~42k beads)

Start from MemPrO's atomistic rank 1 orientation, for example `mempro -f 4G1U.pdb -o 4G1U_Orient` and then `4G1U_Orient/Rank_1/`:

```
$PY footprint_md.py prepare runs/4G1U --pdb 4G1U_Orient/Rank_1/oriented_rank_1.pdb --info 4G1U_Orient/Rank_1/info_rank_1.txt
$PY footprint_md.py build runs/4G1U
$PY footprint_md.py run runs/4G1U/clash --ns 5
$PY footprint_md.py run runs/4G1U/map --ns 5
$PY footprint_md.py analyse runs/4G1U
```

### QmcA (~234k beads; best on a GPU)

The atomistic model in `tests/data/QmcA` is not oriented, so `prepare` fits its coarse-grained model onto MemPrO's rank 1 orientation by the backbone beads. Three of its chain IDs (V, X, Z) are each used for two of its 26 chains, with TER records between them. martinize2 keeps TER-separated chains apart even when they share an ID, so `prepare` writes a TER after every chain: at TER records, at chain ID changes, and where residue numbering restarts. There is no limit on the number of chains. Without the TER records, chains sharing an ID become doubled residues, and martinize2's graph repair then runs for hours. martinize2 takes about 6.5 minutes on QmcA (3 minutes with the speed-ups in `pstansfeld/vermouth-martinize`).

```
Q=../../tests/data/QmcA
$PY footprint_md.py prepare runs/QmcA --pdb $Q/QmcA.pdb.gz --ref $Q/protein-cg_rank_1.pdb.gz --info $Q/info_rank_1.txt
$PY footprint_md.py build runs/QmcA
$PY footprint_md.py run runs/QmcA/clash --ns 5
$PY footprint_md.py run runs/QmcA/map --ns 5
$PY footprint_md.py analyse runs/QmcA
```

For an error bar that includes the random lipid placement, build independent replicates with `build --reps 3`. This makes `clash_r1`, `map_r1`, `clash_r2`, and so on; run each of them, and `analyse` then reports each run and the mean and standard error across replicates. For example, 3 replicates of 20 ns per method, analysed after 5 ns:

```
$PY footprint_md.py build runs/4G1U --reps 3
for d in runs/4G1U/{clash,map}_r{1,2,3}; do $PY footprint_md.py run $d --ns 20; done
$PY footprint_md.py analyse runs/4G1U --skip-ns 5
```

### Differential stress

Insane4MemPrO builds for zero differential stress. To express a mismatch as a stress, run protein-free controls with the same lipids and settings, and pass them to `analyse` as a reference:

```
$PY footprint_md.py control runs/control --reps 3
for d in runs/control/control_r{1,2,3}; do $PY footprint_md.py run $d --ns 20; done
$PY footprint_md.py analyse runs/QmcA --skip-ns 5 --reference runs/control
```

The controls give the reference area per lipid of each leaflet and the area compressibility, K_A = kT⟨A⟩/var(A), from the box-area fluctuations. `analyse` then adds each leaflet's strain against the reference, and the differential stress Σ ≈ (K_A/2)(strain_up − strain_lo), in mN/m (upper minus lower tension; 0 is zero differential stress). K_A from 3 × 20 ns is uncertain by about ±40%, and every Σ scales with it.

`run` skips stages that have already finished (their `.gro` exists) and resumes from a checkpoint (`.cpt`). Pass extra `gmx mdrun` options with `--mdrun-args`, e.g. `--mdrun-args "-nb gpu -ntomp 8"`.

## What each step does

| Step | Does | Writes |
|---|---|---|
| `prepare` | Runs martinize2 on the atomistic model: Martini 3, elastic network (700 kJ mol⁻¹ nm⁻², 0.5–0.9 nm), backbone position restraints, all chains merged into one molecule named `Protein`. Fits onto `--ref` if given. Reads the thickness from MemPrO's `info_rank_N.txt`. | `protein-cg.pdb`, `protein-cg.itp`, `system.json` |
| `build` | Runs Insane4MemPrO with `-footprint clash` and `-footprint map`, with the same box and lipids (default `-l POPE:8 -l POPG:2`), with ions to neutralise and 0.15 M salt. The box is sized as MemPrO `-bd` does: 4 nm beyond the protein in x and y, 5 nm in z. MemPrO's thickness is passed as `-mt`. Writes the topology and an index for the `PROTEIN LIPID SOL_ION` temperature groups. | `clash/`, `map/` |
| `run` | Energy minimisation (`mdp/em.mdp`), 0.5 ns NPT at 10 fs with backbone restraints (`mdp/eq.mdp`), then `--ns` ns NPT at 20 fs (`mdp/md.mdp`). Both NPT stages use c-rescale, semi-isotropic, 1 bar, 310 K. | `em/eq/md.*` |
| `analyse` | Takes the frames after `--skip-ns` (default 1 ns), every 0.1 ns. Assigns lipids to leaflets by phosphate height in the first frame. Counts each leaflet's lipids more than `--cutoff` (3 nm) from any protein bead within 2.5 nm of the midplane, and measures each leaflet's area on its own curved surface (see below). Also runs `mempro-leaflets` on the same frames. | `results.json` and the table below |

Example output:

```
method   n_up   n_lo   rho_up   rho_lo  up-lo (%)    se (%)  leaflets (%)
clash     381    386   1.6549   1.6677      -0.35      0.20         +2.70
map       369    367   1.6999   1.6892      +0.93      0.18         +0.50
```

- `rho`: lipids per nm² of the leaflet's surface beyond the cut-off.

**Curved leaflet areas.** A bilayer that bends, such as the dish that forms inside QmcA's ring, would look like a leaflet mismatch if the leaflets were measured in projection. Their lipids shift sideways in opposite directions, and the concave leaflet seems denser. So in each frame `analyse`:
- fits smooth surfaces to the two phosphate layers (Gaussian, `--smooth` 1 nm) and takes their mean as the midplane;
- assigns each lipid to a region by its foot on the midplane along the local normal;
- measures each leaflet's area on the surface parallel to the midplane at the mean distance of its glycerol beads, near the monolayer's neutral plane.

On a synthetic 2 nm-deep dish whose leaflets are equally packed, this gives a mismatch of about 1%, against 10.6% in projection. A flat membrane gives the same result either way. `--projected` restores the projected measure.
- `up-lo`: the difference between the leaflets in area-weighted packing (the sum of each lipid's area per lipid, per nm²) in that region. It equals the density difference for symmetric compositions, and also works for asymmetric ones. 0 means matched.
- `se`: the standard error of `up-lo` from 1 ns blocks.
- `leaflets`: `mempro-leaflets`' differential strain over the same frames. It uses the footprint definition under test, so it is not independent of the build.

Things to keep in mind when reading the results:

- **What a match means.** A build whose leaflets stay matched in MD has `up-lo` near 0. A built-in mismatch shows up as a persistent `up-lo` of the same sign. A few ns is enough for the far region to equilibrate laterally, but not for any curvature the mismatch induces to develop.
- **Region size.** The region beyond 3 nm must be large enough to give a stable density: at least about 100 nm².
- **QmcA's ring interior.** The middle of the ring (r < ~7 nm) is also more than 3 nm from the protein. Its lipids can't exchange with the outside over a few ns, so `analyse` reports it as a separate `enclosed` region next to the `bulk`.
- **Noise.** Repeat runs (rebuilding gives a different random lipid placement) give a better estimate of the noise than `se` alone.
