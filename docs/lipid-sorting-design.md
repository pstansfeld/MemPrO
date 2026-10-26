# Design: lipid composition around a membrane protein (`-sort`)

Status: design (2026-10-25). Bacterial PE/PG/cardiolipin first; approach 1 (a physical model from the predicted membrane deformation) before any knowledge-based model.

## Problem

Insane4MemPrO gives each leaflet one composition and spreads it uniformly (`assign_lipids`). In MD, lipids sort around the protein:
- anionic lipids (PG, cardiolipin) gather at basic residues near the interface;
- cone-shaped lipids (PE, cardiolipin) gather where the leaflet curves towards its tails;
- lipids whose tails match a locally thinned or thickened bilayer gather there.

In Martini 3 this sorting takes hundreds of ns to µs to reach equilibrium. A build that starts close to the equilibrium distribution would be:
- a prediction in its own right (which lipids a protein enriches, and where);
- a shorter equilibration for production MD.

## Model

A mean-field lateral sorting model, applied per leaflet. Each lipid site at position x gets a probability for each species j:

p_j(x) ∝ λ_j · exp(−βE_j(x))

- λ_j is solved per leaflet so the expected number of each species equals the requested composition. Sorting redistributes lipids; it never changes how many there are.
- E_j(x) has three terms, each computed from what the build already knows:

| Term | Energy per lipid | Inputs |
|---|---|---|
| Hydrophobic mismatch | (K_A/2) · a_j · ((h(x) − h_j)/h_j)² | h(x) = local leaflet hydrophobic thickness, from MemPrOD (`-def`) or flat; h_j = the species' preferred leaflet thickness; a_j = area per lipid; K_A = area compressibility |
| Curvature | −κ · a_j · c0_j · 2H(x) | H(x) = mean curvature of the leaflet's head surface (sign: positive when it curves away from the tails); c0_j = spontaneous curvature; κ = bending modulus |
| Electrostatics | q_j · e · φ(x) | φ(x) = Debye–Hückel potential of the protein's charged beads at the leaflet's head plane; q_j = lipid charge |

Expected sizes, with Martini-like parameters (K_A ≈ 0.3 N/m from our controls, κ ≈ 20 kT, ε_r = 15; Debye–Hückel with ε_r = 15 gives a Bjerrum length of 3.6 nm and a Debye length of 0.35 nm at 0.15 M):
- mismatch: about 0.5 kT for a 15% thickness mismatch;
- curvature: about 0.5 kT for PE versus PG at a curvature radius of 13 nm (the QmcA ring dish);
- electrostatics: about 0.7 kT per lipid charge at 0.7 nm from a lysine or arginine bead and about 2.7 kT at 0.5 nm, so roughly twice that for cardiolipin (−2). The screening length is the least certain part (Martini has explicit ions, so Debye–Hückel is only a guide) and is an option (`-sort_debye`) to calibrate against the referee MD.

For bacterial PE/PG/cardiolipin with natural 16:0/18:1 tails, all three species have the same tail beads in Martini 3. So the mismatch term separates them only through cardiolipin's different packing; curvature and electrostatics do most of the sorting. The mismatch term matters when tail lengths differ (e.g. DLPE/POPE/DPPE, or cardiolipins TMCL/TOCL).

### Parameters, measured rather than assumed

- **h_j, a_j:** from short protein-free Martini 3 bilayers of each species (or the species at 20% in a POPE host for cardiolipin, which is not stable as a pure bilayer). `footprint_md.py control` already does this setup.
- **K_A:** from the same controls (area fluctuations), as `analyse --reference` already does.
- **c0_j:** literature values to start (PE about −0.3 nm⁻¹, PG about 0, cardiolipin about −0.6 nm⁻¹ with Na⁺). These are the least certain parameters; they are reported separately in the validation so a wrong value is visible.
- **κ, ε_r, Debye length:** Martini 3 values. The model is calibrated against Martini, which is also the referee.

### Measured (2026-10-25)

Protein-free Martini 3 lipidome v2 bilayers, 310 K, 0.15 M NaCl, 200 ns (50–200 ns analysed):

| Bilayer | PO4 depth from midplane (nm) | Area per lipid (nm²) | K_A (kT/nm²) |
|---|---|---|---|
| POPE | 2.135 | 0.554 | 64 |
| POPG | 2.000 | 0.608 | 47 |
| POPE:TOCL 80:20 | POPE 2.104, TOCL 2.114 | TOCL ≈ 1.21 (POPE area assumed unchanged) | 59 |

- `SORT_DEPTH` is POPE 2.135, POPG 2.000, TOCL 2.145 (TOCL scaled by its depth relative to POPE in the mixture); `SORT_KA` is 60 kT/nm².
- POPG sits 6% shallower than POPE, so the mismatch term favours PG where the membrane thins. At the 4G1U surface (−8% thickness) that is worth only about 0.1 kT, small next to charge.
- The Insane area table (POPE 0.58, POPG 0.64, TOCL 1.30) is 4–7% larger than these v2 areas. It is left unchanged because `-footprint map` and the zero-stress checks were validated with it.

## Inputs from the deformation

- With `-def`, h(x) and H(x) come from MemPrOD's `Membrane_pos` (midplane and thickness per pixel), smoothed to the 0.1 nm footprint grid that `membrane_geometry` already builds.
- Without `-def`, h(x) is flat and H(x) is zero, and only electrostatics sorts.
- Known limits of MemPrOD, from the MD comparison on 2026-10-25:
  - 4G1U: thinning at the protein surface is about right (−6.5% predicted, −8% in MD), but it decays over about 1 nm instead of 2 nm.
  - 2WIE ring lumen: the midplane rises about right (+0.4 nm predicted, +0.6 nm in MD), but MemPrOD predicts thickening where MD thins by about 0.6 nm.
  - So the mismatch term inherits MemPrOD's errors. Electrostatics does not depend on MemPrOD.

## Implementation in Insane4MemPrO

1. `-sort none|model` (default `none`, so existing builds are unchanged). With `model`, each leaflet's species are assigned to sites by probability instead of uniformly.
2. In `assign_lipids`:
   - compute E_j at each site (per leaflet, at the site's head position);
   - solve λ_j by iterative scaling (Sinkhorn), so that each row (site) sums to 1 and each column (species) sums to its count;
   - sample species site by site from these probabilities, keeping the exact count of each species;
   - keep the existing handling of multi-site lipids (cardiolipin takes two sites).
3. `-footprint map` already fills each leaflet to (free area) / (area per site). With sorting, the local area per lipid changes, but the total lipid area of each leaflet does not, so zero differential stress is kept.
4. The parameter table goes next to the area-per-lipid table, with its source for each value.
5. The build log reports, per leaflet and species, the predicted enrichment within 1 nm of the protein. With an option, the probability maps are written to `.npz` for inspection.

## Validation plan

1. **Referee MD.** A deforming, charged protein in an E. coli-like membrane, POPE:POPG:cardiolipin 75:20:5 (CDL2, −2) in both leaflets, 0.15 M NaCl. 4G1U first: its thinning is measured and MemPrOD predicts it about right. At about 1.6 ns per minute for 32k beads, 1 µs takes about 10 hours on one GPU.
2. **Uniform start against sorted start.** Same box and counts, two replicates each, 1 µs. Measure the enrichment of each species against distance from the protein, and per residue (as in MemProtMD), averaged over the last 500 ns.
3. **Tests:**
   - Does the sorted start match the equilibrated distribution better than the uniform start at t = 0? (prediction quality)
   - Does it reach the equilibrated distribution sooner? (practical value)
   - Which terms carry the prediction? Compare the model with each term switched off.
4. **Outer-membrane protein.** An OMP (e.g. OmpF, 2OMF) in an asymmetric outer membrane: 100% Kdo2-lipid A (`KLA`) in the outer leaflet, POPE:POPG:cardiolipin in the inner leaflet. It tests a thinner, asymmetric membrane: only the inner leaflet has a mixture to sort, and the barrel's short hydrophobic belt thins the membrane around it. Needs the Martini 3 Kdo2-lipid A topology and Ca²⁺ in the outer compartment.
5. **Tail-length test.** The same protein in a POPE/DLPE/DPPE mixture, where mismatch is the only term that separates the species. This tests the MemPrOD-based term directly.

## Results: 4G1U referee (2026-10-26)

Two 1 µs uniform-start runs of 4G1U in POPE:POPG:TOCL 75:20:5 (lipidome v2, 0.15 M NaCl), against 40 Insane builds per condition in the same box. Enrichment is a species' fraction among the 15 lipids nearest the protein in a leaflet, over its fraction in the leaflet. Ranking by distance rather than using fixed distance shells is needed because MD lipids pack closer to the protein than built ones (about 42 against 14 per lower leaflet within 0.7 nm). Errors are standard errors over the two MD replicates or four groups of ten builds.

| Nearest 15 | Upper POPG | Upper TOCL | Lower POPG | Lower TOCL |
|---|---|---|---|---|
| MD, 250–1000 ns | 1.40 ± 0.17 | 1.43 ± 0.25 | 1.61 ± 0.03 | 2.88 ± 0.09 |
| `-sort none` | 0.91 | 0.96 | 1.02 | 0.85 |
| `-sort model`, flat | 0.97 | 0.75 | 1.55 | 1.56 |
| `-def`, charge only | 0.95 | 0.71 | 1.48 | 1.93 |
| `-def`, mismatch only | 0.96 | 0.54 | 1.03 | 0.92 |
| `-def`, curvature only | 1.22 | 1.19 | 1.16 | 1.46 |
| `-def`, all terms | 1.33 ± 0.11 | 1.19 ± 0.07 | 1.68 ± 0.10 | 2.61 ± 0.30 |

- The MD sorting equilibrates within about 100 ns; enrichment in 100–250 ns already matches 750–1000 ns.
- `-sort model -def` reproduces the MD enrichment in both leaflets, within 1–2 standard errors.
- Charge drives the cytoplasmic leaflet: within 2 nm of its head plane 4G1U has 44 Lys/Arg and 16 Asp/Glu. The periplasmic side is net neutral, and its enrichment comes from the curvature term on the MemPrOD-deformed surface, so a flat build misses it.
- Mismatch adds nothing for these lipids (identical tails, about 0.1 kT).
- Debye length: 0.35 nm (Debye–Hückel at 0.15 M) fits. At 0.6–1.5 nm the cytoplasmic leaflet over-enriches (TOCL 4–9×), and the periplasmic leaflet still shows no enrichment without `-def`.
- Cardiolipin, which takes two sites, is under-placed in the 15 nearest and over-placed in the next 15 in all builds, including `-sort none`. Taking the second site next to the protein is harder, which may limit its first-shell enrichment.
- Recommended use: `-sort model -def <MemPrOD Membrane_Data/>`.

### Sorted start (2 × 200 ns)

Two runs started from single `-sort model -def` builds, enrichment of the nearest 15 against time (mean of the two runs; uniform-start runs for comparison):

| Start | Time (ns) | Upper POPG | Upper TOCL | Lower POPG | Lower TOCL |
|---|---|---|---|---|---|
| sorted | 0–5 | 1.65 | 2.04 | 2.51 | 0.68 |
| sorted | 100–200 | 1.38 | 1.30 | 2.12 | 1.32 |
| uniform | 0–5 | 0.99 | 0.68 | 0.67 | 0.68 |
| uniform | 100–200 | 1.23 | 1.59 | 1.75 | 2.84 |
| uniform | 250–1000 | 1.40 | 1.43 | 1.61 | 2.88 |

- POPG starts at the MD level and stays there: no initial 50–100 ns of sorting. The lower leaflet holds slightly above the uniform-start value (2.1 against 1.6–1.75).
- Upper TOCL starts high in single builds and relaxes to the MD value within 50 ns.
- Lower TOCL is not resolved: the leaflet has 14 cardiolipins and the nearest 15 hold 0–3. One run lost its first-shell cardiolipin (0.26), the other gained it (2.39). The builds put it in the second shell (as above), and whether MD then draws it in needs longer or more runs.

## Results: OmpF, inner leaflet (2026-10-26)

One 1 µs run of OmpF (2OMF) with lipid A (LPA3, Ca2+) outside and POPE:POPG:TOCL 75:20:5 inside (17 TOCL), against 40 builds per condition. Inner leaflet, nearest 15 lipids:

| | POPG | TOCL |
|---|---|---|
| MD, 250–1000 ns (three 250 ns blocks) | 1.19 ± 0.09 | 0.88 ± 0.31 |
| `-sort none` | 1.02 | 0.86 |
| `-sort model`, flat | 0.85 | 0.59 |
| `-def`, charge only | 0.85 | 0.58 |
| `-def`, curvature only | 1.02 | 0.55 |
| `-def`, mismatch only | 1.00 | 0.58 |
| `-def`, all terms | 0.95 | 0.17 |

- MD sorting at OmpF is weak: POPG slightly enriched, TOCL not enriched. Its 50–250 ns blocks range from 0.4 to 2.5, so one run does not pin it down.
- The model predicts depletion, and with `-def` strongly so; the terms compound. The 4G1U agreement does not carry over.
- Charge: within 1 nm of the inner head plane OmpF has 12 Lys/Arg against 21 Asp/Glu (27/27 within 2 nm), so the screened potential repels anionic lipids, yet MD shows slight POPG enrichment.
- Deformation: MD inner heads sit 0.45 nm towards the midplane at the protein and recover over about 3 nm; the MemPrOD surface dips 0.31 nm and recovers within 1.2 nm. The sharper MemPrOD dip gives a larger curvature, here of the sign that repels TOCL.

## Found while implementing

- `assign_lipids` took sites in list order, giving each type a block of sites. `-footprint map` appends the sites it adds in gaps next to the protein to the end of the list, so the last type in the composition collected them: in a uniform 4G1U build POPG was 27% within 2 nm of the protein against 18% further out. Sites are now assigned in random order, so `-sort none` is uniform.

## Open questions

- Whether Debye–Hückel with ε_r = 15 matches Martini's electrostatics closely enough. The referee MD settles this; a fitted prefactor is the fallback.
- Cardiolipin's charge state: CDL2 (−2) is the default; CDL1 (−1) is an option.
- Asymmetric bacterial membranes, such as an inner-membrane composition that differs per leaflet, already work: each leaflet is sorted from its own composition.
- Approach 2 (knowledge-based, e.g. MemProtMD contacts) is a later step. It would add a per-residue term near the protein surface.
