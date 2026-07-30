# TSASE

TSASE provides atomistic simulation and saddle-search tools originally
authored by the Henkelman group: <https://theory.cm.utexas.edu/tsase/>.

## Field-coupled SSNEB example

The maintained example is a self-contained PbZrO3 AFE-to-R3c transition under
an electric field along the crystallographic `[-1, -1, -1]` direction. Its
configuration, structures, model, preprocessing command, and run instructions
are documented in [`example/README.md`](example/README.md).

The example uses `example/models/PZO.model` directly in MACE-Field mode, with
no external species-charge map, and writes to
`example/runs/pzo_neg111_field_ssneb/`.
