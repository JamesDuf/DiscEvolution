# Nibi EOS comparison

240 disc runs: 120 parameter combinations for each of `DeadZoneEOS` and
`IrradiatedEOS`. Every disc carries the same 35 initial embryos (8,400 tracks).
These are model growth/migration tracks, not an N-body stability calculation.

- Disc mass: 0.05, 0.075, 0.1, 0.125, 0.15 solar masses.
- Initial accretion rate: 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6 solar masses/year.
- Characteristic radius (`Rd`): 50, 100, 150, 200 AU.
- Gamma: 0.375; star: 1 solar mass, 2.5 solar radii, 4000 K.
- Grid: 1,000 cells, natural spacing, 0.1 to 1,000 AU; final time: 3 Myr.
- Embryos: 0.1 Earth masses, zero initial envelopes, all inserted at time zero.
- Starting radii: 0.2 to 2.0 AU in 0.1 AU steps, 3 to 10 AU in 1 AU steps,
  then 15 to 50 AU in 5 AU steps. Endpoints are not duplicated.
- DeadZoneEOS: alpha_SS_dead=1e-5, alpha_SS_active=1e-3, CR ionization.
- IrradiatedEOS: uniform alpha_SS=1e-5. The driver's wind solver fits the
  specified accretion rate separately for each run.

Other settings, including chemistry, planetesimals and torque-map output, are
inherited from `config.json`. Fields unused by the chosen EOS may be null.
The base templates intentionally leave the three swept disc parameters null;
run generated configs, not the templates directly.

## Generate

From this directory, run `python generate_grid.py`. This rewrites generated
configs, `Irradiated_config.json` and `manifest.txt`; edit `config.json` for
shared settings and the generator for grid/planet choices. Do not regenerate
or edit these files while jobs are queued or running.

Manifest indices 0-119 are DZ, and 120-239 are Irradiated. Index `i+120` is the
matched Irradiated counterpart of DZ index `i`.

## Transfer from the Mac

If these files have not been committed and pushed, copy this directory:

```bash
rsync -av --exclude=output --exclude=logs --exclude=__pycache__ \
  /Users/james/Documents/GitHub/DiscEvolution/james/october2026/NibiPopSynthOct11/ \
  jdufr@nibi.alliancecan.ca:~/projects/def-mbalogh/jdufr/DiscEvolution/james/october2026/NibiPopSynthOct11/
```

The parent repository and installed DiscEvolution code must also be current on
Nibi. This transfer only updates this experiment directory.
The multiple-dead-zone warning filter is in `DiscEvolution/eos.py`, so that
file must also be synchronized (or committed, pushed and pulled) before running.

Multiple-dead-zone diagnostics use logarithmic depth `-log10(min(Lambda))`
below the Elsasser threshold of one. A warning is emitted only if a secondary
zone is at least 50% as deep as the main zone, and at most once per EOS
instance. This changes logging only; the deepest zone still determines the
model's dead-zone radius. With this batch script the warning goes to `.err`.

## Submit on Nibi

```bash
cd ~/projects/def-mbalogh/jdufr/DiscEvolution/james/october2026/NibiPopSynthOct11
mkdir -p ~/projects/def-mbalogh/jdufr/output/logs
sbatch --array=0,120 run_array.sbatch
```

This first submits only one matched pair, still at the full requested 3 Myr
duration. The script requests 1 CPU, 8 GB memory, and 12 hours per task. These
are provisional resources: inspect the pilot's `seff JOB_ID` output and logs
before submitting the remaining tasks. Resource requests can be overridden
with `sbatch --mem=... --time=...`.

Create the project output's `logs` directory before submitting: Slurm opens
the `.out` and `.err` files before the batch script starts. The log directives
use absolute paths because shell variables and `~` are not expanded there.

After the pilot succeeds, submit the remaining 238 tasks (at most 120 at once):

```bash
sbatch --array=1-119,121-239%120 run_array.sbatch
squeue -u "$USER"
```

To submit all 240 from scratch, use `sbatch run_array.sbatch` instead.
The concurrency cap is 120 simulations, each using one CPU and 8 GB RAM.
Actual concurrency depends on available resources, account priority and Slurm
limits. As an illustration, 240 four-hour runs at a sustained concurrency of
120 would take about eight hours, excluding queue delays. This is not a runtime
prediction; use the pilot measurements. Increasing CPUs per task alone does
not parallelize this driver. The 12-hour wall-time request is a per-task limit,
not an overnight completion guarantee.
Each task has independent logs under
`~/projects/def-mbalogh/jdufr/output/logs/ARRAY_JOB_ID_TASK_ID.out` and `.err`,
and a separate output directory
under `~/projects/def-mbalogh/jdufr/output/CONFIG_NAME/ARRAY_JOB_ID_TASK_ID/`, containing its config and HDF5
file. Re-submissions create new output directories and do not resume old runs.
The batch script supplies this project-level location with `--output_dir`,
overriding the relative `output_dir` in the generated JSON. Direct Python runs
without that override still use the JSON path relative to the working directory.

The batch script loads Python and activates `~/discevolution-env`. It disables
the driver's predictive ETA cutoff using `--eta_abort_hours inf`; Slurm still
enforces the wall-time limit. Direct runs of the configs retain the template's
4-hour predicted-runtime cutoff. Early exits now set `complete=false` and
raise an error; successful outputs also record `final_time_yr`. Check that
outputs reached 3,000,000 years, not merely that an HDF5 file exists.

Slurm array syntax reference: https://slurm.schedmd.com/job_array.html
