# pinn_mgn_pytorch

PyTorch companion to the MATLAB PI-MGN code in `../pi_fem_mgn_successful_one_batch_test_Jan_30/`.
Everything is benchmarked against its own vectorised Q4 FEA (a port of the MATLAB `FEM/` folder).

```
elastic_sim/
  meshes.py    structured Q4 meshes: rectangle/beam, quarter plate with central hole
  fem.py       baseline FEA: plane-stress Q4, body force, edge tractions, Dirichlet, timing
  problems.py  benchmark cases (Timoshenko beam w/ analytic solution, UDL cantilever,
               plate with hole uniaxial/biaxial, gravity-loaded square) for FEA + PINN
  pinn.py      PINN: strong-form or energy loss, hard Dirichlet, input/output normalisation,
               Sobol collocation, Adam + cosine annealing, L-BFGS, ensembles
  pool.py      training pool of FEA-labelled graphs with mesh metadata (192 meshes)
  mgn.py       MeshGraphNet (LayerNorm | BatchNorm | none), feature normalisers, amplitude head,
               physics residual loss, ensembles, coarse->fine warm-start curriculum
  fsi.py       Stokes channel flow + elastic flap: differentiable GNN FSI solver
scripts/
  run_pinn_benchmark.py   PINN ensemble vs FEA on 5 cases
  run_mgn_benchmark.py    MGN ensemble accuracy on held-out meshes + timing vs FEA / PCG
  run_fsi.py              FSI GNN (fluid-only warm start -> coupled), unseen flap geometries
results/                  JSON metrics, figures, trained weights
```

Setup (Python 3.13, Apple Silicon MPS or CPU):

```
python3 -m venv ../.venv && ../.venv/bin/pip install torch numpy scipy matplotlib
../.venv/bin/python scripts/run_pinn_benchmark.py --seeds 3
../.venv/bin/python scripts/run_mgn_benchmark.py --seeds 3 --epochs 300 --hidden 64 --layers 12 --curriculum
../.venv/bin/python scripts/run_fsi.py --epochs 200
```

See `../RESEARCH_SUMMARY.md` for the results.
