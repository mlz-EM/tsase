# PbZrO3 field-SSNEB example

This self-contained example is based on the F3 film calculation. It connects
the bundled AFE and R3c structures with 22 images, remeshes to 77 images, and
applies an electric field of magnitude `0.003` along the crystallographic
`[-1, -1, -1]` direction.

All paths in `input.yaml` are relative to this directory. The endpoint
structures are under `structures/`, the model is `models/PZO.model`,
preprocessing writes to `preprocessed/`, and the SSNEB run writes to
`runs/pzo_neg111_field_ssneb/`. The existing F3 results, when present, live at
that same run path.

The MACE-Field calculator applies the field internally and reports its own
polarization. It therefore does not use a `model.charges` species map; fixed
species charges are only needed by calculators that use TSASE's external
enthalpy wrapper.

## Requirements

Create an environment containing TSASE's Python dependencies, MACE with
MACE-Field support, PyTorch, ASE, NumPy, SciPy, PyYAML, and Matplotlib. The
bundled configuration selects CUDA and double precision by default.

Run the commands below from the repository root.

## 1. Preprocess the endpoints

```bash
python example/preprocess.py
```

This relaxes the two endpoints and creates `example/preprocessed/run.yaml`.

## 2. Run SSNEB

For one GPU:

```bash
python example/run.py
```

For an MPI or scheduler launch with one rank per image worker, invoke the same
script through the site's launcher; `--parallel auto` enables image-parallel
execution whenever more than one rank is detected.

Useful command-line overrides include:

```bash
python example/run.py --device cpu --parallel false
python example/run.py --max-steps 10 --fmax 0.05
python example/run.py --output-dir /path/to/a/run
```

The first two commands retain the bundled relative paths. An explicit
`--output-dir` may be absolute or relative to the current working directory.
