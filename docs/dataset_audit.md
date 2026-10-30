# Dataset audit (28 October 2026)

## Bottom line

**No real radiation dataset was available locally, and none was downloaded for this draft.**
All experiments use a clearly labelled synthetic six-window Poisson generator. This is a
Gate 1 partial failure in the research plan's terms: stream generation is verified, but real
data access, format, labels, and licensing were not verified end-to-end.

The synthetic study is scientifically appropriate for testing the *mechanism* (guarded
adaptation, drift response, fault resilience) under controlled conditions. It is **not**
evidence of performance on real portal-monitor data, and it is not called real-world
validation anywhere in this repository.

## What was inspected locally

| Path | Finding |
|---|---|
| `..\SIH26166-Lunar-Image-Correspondence\` | Unrelated lunar image-correspondence project. The directory is an empty cloud/reparse placeholder in this workspace. No radiation data. |
| `..\test_rf_integration.py` | Unrelated stray test for a lunar-image random-forest model. Not imported or modified. |
| `..\` workspace | No RADAI, PyRIID, JRC, or portal-monitor count files. |
| Downloads folder | No radiation count stream. The `ch2_ohr_ncp_*.zip` archives are Chandrayaan-2 lunar images, not radiation data. |

No file outside `rad-daes/` was modified.

## Dataset candidates from the research plan

### RADAI

**Verified facts (primary sources):**

- RADAI is a large-scale **synthetic** dataset produced by ORNL/LBNL/LANL/NCSU for
  radiation detection algorithm development. It integrates Monte Carlo simulations with
  urban scenarios and models NORM, people, vehicles, urban clutter, cosmic-ray and
  rain-induced transients. It provides **list-mode** detector data with motion and response
  modelling, three complementary datasets (training, developer, testing), an online scoring
  portal, and an open software toolkit. Source: Ghawaly et al., *IEEE Transactions on
  Nuclear Science* 73(5):2045–2066, 2026, DOI `10.1109/TNS.2026.3682654`.
- The earlier public urban dataset is distributed as a 13 GB gzip-compressed tarball
  (29 GB uncompressed) under CC BY 4.0. Each CSV file has two columns: time since the
  previous detection event in microseconds, and detected photon energy in keV. Access
  requires Globus. Source: Ghawaly et al., *Scientific Data* 7:328, 2020,
  DOI `10.1038/s41597-020-00672-2`.
- The 2026 RADAI release is accessed through the Berkeley Data Cloud (BDC); the public wiki
  states that registration/login is required and that the data are CC BY 4.0. Source:
  <https://bdc.lbl.gov/wiki/public/radai-interactive-datasets/> (accessed 28 Oct 2026).

**Why it was not used:** access requires a BDC account and a large download; no credentials
were used and no data were downloaded. The list-mode format would also require a separate
energy-calibration and windowing step, which was not verified in this draft.

**What would be needed:** a BDC account, the RADAI developer split, the RADAI toolkit, a
documented energy-calibration/windowing procedure, and a check that the selected scenarios
contain the required temporal background and source labels. None of that is claimed here.

### PyRIID

**Verified facts (primary source):**

- PyRIID (`riid`) is a Python package from Sandia National Laboratories for gamma-spectrum
  synthesis, model training, and visualization. It provides `PassbySynthesizer` (sequences
  of spectra for a detector moving past a source) and `StaticSynthesizer` (static spectra
  with configurable live time and SNR, optional Poisson noise). Source:
  <https://sandialabs.github.io/PyRIID/> and the `riid.data.synthetic` API documentation
  (accessed 28 Oct 2026).
- The documented supported Python versions are 3.10–3.12. This environment is Python 3.14.7.

**Why it was not used:** PyRIID was not installed or executed. Its pass-by output is a
sequence of spectra, not the six-window 1 Hz stream required here; converting it would
require a documented windowing and calibration step. The Python-version mismatch is a
practical blocker for a same-day install. No PyRIID output is claimed.

### JRC UPu-Gamma

**Verified facts (primary source):**

- The JRC Data Catalogue hosts uranium and plutonium gamma-ray spectra measured with
  LaBr3, CdZnTe, and HPGe/ORTEC detectors, intended for the development of data-evaluation
  programmes. The collection is described at
  <https://data.jrc.ec.europa.eu/collection/id-00266>; the associated report is JRC124107
  (Zsigrai et al., 2020).
- These are **reference spectra of certified materials**, not continuous portal-monitor
  time series. They do not supply the temporal background/source/fault labels needed here.

**Why it was not used:** no continuous temporal stream, no portal background labels, and no
verified six-window mapping. The collection is not a substitute for a time-series benchmark.

## Synthetic generator (what was actually used)

`src/rad_daes/data/synthetic.py` generates a 6-hour, 1 Hz, six-window stream. The generator
is deterministic given a seed and writes the following labels:

- `mu_true[t, j]`: noise-free background mean per window;
- `source[t, j]`: source excess per window;
- `y[t]`: source-event label;
- `fault[t]`: fault mode (`none`, `dropout`, `jitter`, `stuck`, `spike`);
- `event_id[t]`: source-event identifier (0 = none);
- `t_obs[t]`: observed timestamp.

Background regimes:

| Regime | Construction | Purpose |
|---|---|---|
| `stable` | constant nominal mean plus small correlated noise | calibration and false-alarm baseline |
| `gradual` | slow sinusoidal + linear trend | adaptation to smooth drift |
| `abrupt` | two step changes in selected windows | recovery after a discontinuity |
| `recurring` | repeated rain/radon-like bumps in W1/W6 | recurring, non-monotone drift; held out from tuning |

Source events:

- Three templates (`cs137`, `co60`, `pu239`) with relative per-window amplitudes.
- 6 events per stream, 20–60 s each, separated by 900–2700 s.
- Excess amplitude drawn from 3–8 baseline standard deviations, so some events are near the
  detection limit.
- Events are injected independently of faults; a fault can overlap an event.

Fault modes (applied to the raw observation after source injection):

| Mode | Implementation | Intended gate response |
|---|---|---|
| `dropout` | set 1–3 channels to NaN | validity rejects |
| `jitter` | timestamp offset outside [0.5, 1.5] s | timing rejects |
| `stuck` | repeat the previous vector for 5–30 samples | Fano ≈ 0 rejects |
| `spike` | add 10–50 baseline σ to one channel for one sample | Fano > 3.5 rejects |

The fault rate is the fraction of samples selected for corruption. When a sample is selected,
one mode is drawn uniformly from the enabled modes. The mixed conditions use rates 5%, 10%,
and 15%; the single-mode conditions use 10%.

## Licensing and redistribution

- No RADAI, PyRIID, or JRC data or code is redistributed in this repository.
- The synthetic generator is original code for this draft.
- The RADAI and JRC sources are cited, not copied.
- The original DAES snapshot is external and not vendored; see `docs/daes_interface.md`.

## What this audit does not establish

- That the six windows are physically meaningful for any particular detector.
- That the nominal count rates correspond to a real portal.
- That the synthetic drift, source, or fault distributions are realistic.
- That the results transfer to RADAI, PyRIID, JRC, or a deployed portal.

These are explicit limitations, not hidden assumptions.
