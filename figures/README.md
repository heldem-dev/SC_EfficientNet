# Figure generation

The PNG files in this directory are the exported publication figures.

`make_revision_figures.py` regenerates Figures 4–11 from the exported CSV/JSON results under `results/`.

`make_xai_figures.py` regenerates Figures 12–13 from the archived XAI result package under `results/xai/` and `xai_class_summary.csv`. Figure 12 shows one correctly classified held-out image per class; the selected files are listed in the script.

Figure 1 and Figure 3 are retained as publication-ready exported graphics. The manuscript-construction scripts are not included in this repository.

Run from the repository root:

```bash
python figures/make_revision_figures.py
python figures/make_xai_figures.py
```
