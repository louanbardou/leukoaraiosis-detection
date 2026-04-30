# Project State — Complete Documentation
**Written:** 2026-04-10  

### What the ABCD Study Is

The **Adolescent Brain Cognitive Development (ABCD) Study** is the largest long-term study of brain development in the United States. It tracks ~11,900 children starting at age 9–10 across 7 annual visits (sessions), collecting neuroimaging, cognitive assessments, mental and physical health data, genetics, and rich environmental variables at 21 research sites nationwide.

### Release 6.0 — The Main Release

Path: `/wynton/group/abcd/6.0/`

**Master data dictionary:** `datadictionary.csv`
- 59.8 MB, 92,419 rows
- Columns: `id`, `domain`, `table_label`, `source`, `table_name`, `name`, `description`, `type_level`, `type_data`, `type_var`
- One row per variable, describing every variable in the entire release

**Tabulated data:** `tabulated/` — 718 files = 239 tables × 3 formats

Every table exists in three identical formats:
- `.tsv` — tab-separated text, opens in pandas, Excel, etc.
- `.parquet` — compressed columnar binary, much faster for code (preferred)
- `.json` — metadata sidecar: per-variable descriptions, value labels, units, whether it is a raw or derived variable

Every table has two index columns: `participant_id` (format: `sub-XXXXXXXX`) and `session_id` (format: `ses-00A` through `ses-06A`).

**Missing/special values:**
- `n/a` — item not applicable at this time point (not administered)
- `555` — question not asked in this wave/version
- `888` — not yet assessed
- `777` — don't know
- `999` — refused

**Study scale:** 11,868 unique subjects × 7 sessions = up to 83,076 rows per table (in practice fewer because not all subjects have all sessions).

### The 239 Tables — By Domain

The table naming convention is `{domain}_{respondent}_{instrument}`:
- `p` = parent/caregiver report
- `y` = youth self-report
- `l` = linked/geocoded (not self-report; derived from residential address)
- `t` = teacher report
- `e` = examiner/rater
- `g` = general/administrative

**`ab` — ABCD Administrative (5 tables)**  
Core study metadata. `ab_p_demo` (236 vars): child age, sex, race/ethnicity, country of origin for child and four grandparents, parental education, household income, insurance, marital status, religiosity, language, adoption history, household roster. `ab_g_stc`/`ab_g_dyn`: site ID, family ID, twin status, scanner model, study flags.

**`covid` — COVID-19 (9 tables)**  
Parent and youth pandemic questionnaires (295 and 311 variables respectively) plus 6 geocoded administrative datasets linked to participants' home addresses: JHU county-level case counts (350 vars), CDC policy surveillance (711 vars), SafeGraph social distancing metrics (350 vars), Census data, BLS labor statistics.

**`ecb` — Endocannabinoid Substudy (1 table)**  
14 variables; collected on a subset of participants.

**`fc` — Family & Culture (30 tables)**  
Covers: family environment scale (parent + youth), acculturation, ethnic identity, neighborhood safety and collective efficacy, parental knowledge and monitoring, school grades and attendance, peer behavior, prosocial behavior, values, unfair treatment experiences, neglectful behavior, pet ownership, driving, multidimensional neglect, problem solving.

**`gn` — Genetics (3 tables)**  
Twin zygosity ratings (25 vars), pairwise genetic relatedness estimates (14 vars), and top principal components of ancestry from genotyping array (32 vars — used as covariates in genetic association analyses).

**`irma` — Hurricane Irma (2 tables)**  
128 and 114 variables; administered only to participants in affected regions; covers exposure, displacement, stress, and impact.

**`le` — Life Events & Linked Environment (69 tables)**  
The broadest environmental domain. All are geocoded datasets linked to participants' residential addresses — not self-report. Sources include EPA, Census Bureau (ACS), CDC, academic research products, and commercial vendors. Covers:
- Deprivation indices: ADI (57 vars), COI (261 vars), SVI (108 vars), MHSVI (54 vars), ICE (6 vars)
- Racial segregation: dissimilarity index, entropy index, exposure/interaction indices
- Crime: ICPSR crime rates (24 vars), Getis-Ord hotspot statistics (24 vars)
- Air quality: NO2, O3, PM2.5, PM10, particulate matter, EJScreen toxics (satellite-based, 9–45 vars each)
- Climate: temperature (PRISM, 21 vars), vapor pressure deficit (21 vars), noise (27 vars)
- Land use: NLCD land cover and tree canopy (90 vars), walkability (EPA), urban/rural, satellite land-use measures
- Education: SEDA school achievement scores for math and reading at 4 geographic levels (county, district, commuting zone, metro); each level has ~105 variables covering test score distributions
- Drug policy: cannabis legalization status, medical marijuana policy, naloxone policy, PDMP status, co-prescribing policy, CDC opioid dispensing rates
- Economic: job counts and density (LODES), social mobility (Opportunity Atlas), rent/mortgage statistics, opportunity zones
- Social services: parks, arts/sports orgs, religious/civic orgs, social service facilities
- Healthcare access: ACA Medicaid expansion status, PLACES behavioral health indicators (84 vars)
- Other: immigration bias measures, lead risk, road proximity, traffic density, elevation, census return rates

**`mh` — Mental Health (58 tables)**  
The deepest clinical domain. Covers:
- Behavior checklists: CBCL (184 vars, parent), ABCL (215), ASR (187), YSR (177, youth), Brief Problem Monitor (parent and youth)
- Temperament/affect: EATQ (91), BIS/BAS (36), UPPS-P impulsivity (33), emotion regulation (13, 40, 74)
- Psychiatric history: family history (1,453 vars — the largest mental health table), KSADS background items
- Life events: parent (222 vars) and youth (218 vars)
- Prodromal/mania: prodromal psychosis scale (73), GBI mania (15), 7-Up mania (12)
- Social: social responsiveness (16), peer experiences (33), cyberbullying (16), resilience (11)
- KSADS diagnostic modules — structured clinical interview covering 21 psychiatric conditions, each with present and past symptom items, and present/past/partial-remission diagnoses:
  ADHD, agoraphobia, autism spectrum, bipolar, conduct disorder, depression (MDD + PDD), DMDD, eating disorders, GAD, homicidality, OCD, ODD, panic, specific phobia, psychosis, PTSD, separation anxiety, sleep problems, social anxiety, suicidality, tic disorders
  — available from both parent and youth reporters (youth covers a subset of modules)

**`mr` — MRI (420 tables)**  
The largest domain by table count. All respondent-coded `y` (youth), all brain imaging.

*Administration:* scanner and site info (7 vars), scanning checklist (21), pre/post-scan questionnaires (48).

*Quality control:* 17 QC tables covering raw data quality (T1w, T2w, dMRI, rsfMRI, three task fMRI runs) and post-processing quality (automated and manual ratings for dMRI, fMRI, FreeSurfer surface reconstruction).

*Structural MRI (sMRI):* FreeSurfer-derived regional brain morphometry. Measures: volume, cortical thickness, surface area, sulcal depth, T1w intensity, T2w intensity, gray/white matter contrast. Parcellations: Desikan atlas (~71 cortical ROIs), Destrieux atlas (~151 ROIs), Fuzzy clustering atlas, and ASEG subcortical segmentation (~30 regions). Every measure × every atlas = a separate table. Subcortical regions include bilateral: accumbens, amygdala, caudate, cerebellum cortex and white matter, hippocampus, pallidum, putamen, thalamus, ventricles, brainstem.

*Diffusion MRI — DTI:* Two acquisition variants (full shell = multi-shell, inner shell = single shell). Four diffusion metrics: FA (fractional anisotropy), LD (longitudinal/axial diffusivity), MD (mean diffusivity), TD (transverse/radial diffusivity). Tissue parcellations: ASEG subcortical, AtlasTrack white matter tracts, and gray matter / gray-white contrast / white matter × Desikan and Destrieux. Total: ~72 DTI tables (36 per shell variant). FA is the key metric for white matter integrity — decreases in leukoaraiosis.

*Diffusion MRI — RSI (Restriction Spectrum Imaging):* Multi-compartment diffusion model decomposing signal into free water, hindered diffusion, and restricted diffusion compartments. Metrics: `fni` (free normalized isotropic), `hnd/hni/hnt` (hindered directional/isotropic/total), `rnd/rni/rnt` (restricted directional/isotropic/total). Same tissue parcellations as DTI. ~56 tables. RSI metrics are more specific to axonal density and myelin than conventional DTI.

*Resting-state fMRI:* BOLD variance by region (Desikan, Destrieux, ASEG, Gordon parcels) and Gordon network correlation matrices (network × network and network × subcortical ROI). Gordon parcels are 333 cortical regions grouped into 12 functional networks.

*Task fMRI — three paradigms:*
- **MID (Monetary Incentive Delay):** Reward anticipation and receipt. Contrasts cover anticipation of large/small/neutral reward and loss conditions vs. fixation, and various reward-loss combinations. ~60 contrast tables × 3 parcellations (ASEG, Desikan, Destrieux) × 2 runs + combined. Behavioral data (reaction times, accuracy) in `mr_y_tfmri__mid__beh`.
- **N-Back:** Working memory (0-back vs. 2-back) and emotional face recognition (positive/negative faces vs. neutral). Contrasts: 0b, 2b, 2bv0b, emotional faces, face vs. place, positive/negative face vs. neutral, etc. Also a recognition memory table. ~70+ tables. Behavioral data in `mr_y_tfmri__nback__beh`.
- **SST (Stop Signal Task):** Response inhibition. Contrasts: correct stop vs. correct go, incorrect stop vs. correct go, any stop vs. correct go, correct go vs. fixation, and incorrect go variants. ~60 tables. Behavioral data in `mr_y_tfmri__sst__beh`.

*MR Spectroscopy:* Two tables — 2D J-resolved spectroscopy and HERMES sequence (edited spectroscopy for GABA and Glx/glutamate quantification).

**`mrs` — MR Spectroscopy (2 tables)**  
See above.

**`nc` — Neurocognition (15 tables)**  
Youth-administered cognitive battery. Key instruments:
- NIH Toolbox (163 vars): fluid and crystallized composite scores + 7 individual tests (Picture Vocabulary, Oral Reading Recognition, Pattern Comparison Processing Speed, List Sorting Working Memory, Dimensional Change Card Sort, Flanker Inhibitory Control, Picture Sequence Memory). Normed standard scores (uncorrected, age-corrected, fully-corrected).
- RAVLT (35 vars): verbal learning and memory across 5 learning trials + delayed recall
- NIH Toolbox Flanker (also in NC): response inhibition
- Delay Discounting (36 vars): impulsive choice / temporal discounting
- WISC-V Matrix Reasoning (36 vars): non-verbal fluid intelligence
- Game of Dice (26 vars): risk-taking behavior
- Social Influence Task (26 vars): peer influence on decision-making
- Emotional Stroop Task (57 vars): attention and emotional interference
- Little Man Task (65 vars): mental rotation / spatial cognition
- Stanford Math Response Time Evaluation (66 vars): arithmetic processing speed
- Cash Choice Task (3 vars): short delay discounting
- Edinburgh Handedness Inventory (8 vars): laterality
- Snellen Vision Screener (6 vars): visual acuity (used as covariate)
- Behavioral Indicator of Resiliency to Distress (34 vars)
- Barkley EF Scale (26 vars, parent-reported): executive function ratings

**`nt` — Technology & Screen Time (15 tables)**  
- **EARS (Ecological momentary assessment of digital media):** Passive phone sensing. App usage statistics (319 app categories), keyboard input statistics (1,375 variables — typing patterns, frequency, content categories), plus pre/post assessment surveys. Provides objective, real-world digital behavior data.
- **Fitbit:** Physical activity (steps, heart rate zones, active minutes) and sleep data, collected with wearable sensors during study visits.
- **Screen Time Questionnaire:** Parent (49–52 vars) and youth (181 vars) self-report of recreational screen time by device type and content.

**`ph` — Physical Health (28 tables)**  
- Anthropometrics: height, weight, BMI, waist circumference (parent and youth report)
- Blood draw (56 vars): complete blood count, comprehensive metabolic panel, lipid panel, HbA1c, CRP
- Blood pressure (29 vars): systolic/diastolic × multiple readings
- Pubertal development: Pubertal Development Scale (parent + youth), Menstrual Cycle Survey; pubertal hormone saliva analysis (47 vars): testosterone, DHEA, estradiol
- Developmental history (242 vars, parent): prenatal exposure (alcohol, tobacco, drugs, medications, stress), delivery complications, neonatal period, early developmental milestones, breastfeeding history
- Medical history (144 vars): chronic conditions, hospitalizations, surgeries, head injuries
- Traumatic brain injury screen (85 vars)
- Medications inventory (393 vars parent, 213 vars youth): all current medications by class
- Nutrition: Block Kids Food Screener (parent 209 vars, youth 211 vars), child nutrition assessment, breastfeeding
- Sleep: Munich Chronotype Questionnaire (119 vars), Sleep Disturbance Scale (29 vars)
- Physical activity: IPAQ (18 vars, parent), YRBSS physical activity (6 vars, youth), Sports/Activities Involvement (836 vars parent)
- Pain questionnaire (83 vars), Respiratory functioning (27 vars)
- Sexual behavior, orientation, and communication (parent 9 vars, youth 74 vars)
- COVID annual survey (parent 155 vars, youth 48 vars)

**`sdev` — Social Development (15 tables)**  
- Parenting: Alabama Parenting Questionnaire (parent 86 vars, youth 110 vars) — positive parenting, inconsistent discipline, harsh discipline, supervision
- Personality: Big Five-based disposition scale (parent 102 vars, youth 242 vars)
- Delinquency: Reported Delinquency Scale (parent 202 vars, youth 356 vars) — property crime, violent crime, drug-related offenses
- Victimization (parent 1,681 vars, youth 1,675 vars) — the two largest tables in the entire dataset; comprehensive trauma and adversity assessment across many event types and contexts
- Emotion regulation: Difficulties in Emotion Regulation Scale (parent + youth, 74 vars each)
- Peer behavior (36 vars): delinquent peer association
- Firearms access and safety (parent BRFSS 8 vars, youth YRBSS 9 vars)
- Neighborhood perception (52 vars)
- Visit type (15 vars): in-person vs. remote assessment tracking

**`su` — Substance Use (46 tables)**  
The most comprehensive substance use battery in a pediatric longitudinal study.

Biological toxicology (objective, biospecimen-based):
- Urine toxicology (112 vars): screens for THC, opioids, cocaine metabolites, amphetamines, benzodiazepines, etc.
- Hair toxicology (119 vars): 3-month retrospective window for cannabis, cocaine, opioids, methamphetamine
- Oral fluid / saliva toxicology (81 vars): recent use window (~hours to days)
- Alcohol toxicology (36 vars): breath and/or blood alcohol, PEth
- Nicotine toxicology (30 vars): cotinine assay

KSADS diagnostic modules:
- Alcohol Use Disorder (parent and youth, 36 vars each)
- Drug Use Disorders (parent and youth, 359 vars each — covers cannabis, cocaine, opioids, stimulants, sedatives, hallucinogens, inhalants, and polysubstance)

Substance-specific questionnaires (youth self-report):
- Alcohol: expectancies (14), motives (23), hangover symptoms (34), problem index RAPI (41), subjective response/effects (28)
- Cannabis: expectancies (13), motives (28), problem index MAPI (23), subjective response (20), withdrawal (20)
- Nicotine: dependence (13), subjective response (21), vaping expectancies (15), vaping motives (12), reasons for ENDS use (27), cigarette expectancies (16), tobacco motives (22)

Use frequency and general instruments:
- Substance Use Interview (836 vars): comprehensive use onset, frequency, quantity for all substances
- Timeline Followback Interview (883 vars): the two largest substance use tables; day-by-day retrospective use calendar for alcohol, cannabis, tobacco, and other drugs over a 90-day window
- Mid-Year Phone Interview (133 vars): between-visit use check
- Low Level Use Questionnaire (51 vars): sensitive screening for very early/minimal use
- Caffeine Use Questionnaire (90 vars)
- Opportunity to Use, Peer Tolerance SU, Perceived Harm, Peer Deviance SU, Sibling Use — social context of use
- Participant Last Use Survey (parent 103 vars, youth 148 vars): day-of-visit use within 24–96 hours
- Community Risk and Protective Factors (parent 16 vars, youth 21 vars)
- Parental rules on substance use (16 vars)
- Substance storage, density, and exposure in the home (381 vars, parent)
- PATH Intention to Use (18 vars)

### Release 6.1 — Incremental Update

Path: `/wynton/group/abcd/6.1/`

Contains two subdirectories:

**`tabulated/`** — the same 239 tables × 3 formats (718 files), updated with corrected and additional data. December 2024 update timestamps.

**`concat/`** — pre-assembled imaging matrices. Instead of individual per-subject files, these are single large `.mat` files where each matrix is subjects × voxels/vertices for a given measure. This makes loading imaging data into Python dramatically faster (one file load instead of 808 individual reads).

`concat/vertexwise/` — cortical surface data:
- `dti/`: FA and longitudinal diffusivity on the cortical surface. 4 smoothing levels (sm0 = unsmoothed, sm16 = 16mm FWHM, sm256, sm1000) × 2 hemispheres (lh, rh) × 2 tissue types (gray matter, white matter) = 16 files per metric. Plus `vol_info.mat` (subject/session index).
- `rsi/`: RSI metrics (fni, hnt, and others) same structure.
- `smri/`: cortical area, thickness, volume. Same smoothing × hemisphere structure.

`concat/voxelwise/` — volumetric diffusion data:
- `dti/`: `fa.mat`, `md.mat` + `vol_info.mat`
- `rsi/`: 18 metric files — fi, fni, hdf, hd, hif, hi, hnd, hni, hnt, ht, rd, rdf, rif, ri, rnd, rni, rnt, rt — plus `vol_info.mat`

RSI metric nomenclature: prefix `f`=free, `h`=hindered, `r`=restricted; suffix `n`=normalized, `d`=directional, `i`=isotropic, `t`=total, `f`=free (within component).

### Imaging Derivatives

Path: `/wynton/group/abcd/6.0/imaging/derivatives/`

**`mproc/` — Minimally Processed MRI (808 subjects)**  
Per-subject directory: `sub-XXXXXXXX/ses-XXXX/{anat/, dwi/}`. Contains FreeSurfer surface reconstructions, T1w/T2w volumetric outputs in MNI space, and diffusion preprocessing outputs. These are the inputs to further analysis pipelines.

**`mrtrix/` — MRtrix Tractography (very partial: 3 subjects)**  
Early pilot. Per subject contains tractography intersection files for specific white matter tracts: BNST (bed nucleus stria terminalis, left and right) and SCC (subgenual cingulate cortex, left and right), plus MNI registration. Not suitable for group-level analysis yet.

**`tractointersect/` — Tract-ROI Intersection Matrices (626 subjects)**  
Available at sessions ses-00A, ses-02A, ses-04A, ses-06A (every other visit). Per subject/session contains matrices of intersection between white matter tracts and cortical/subcortical ROIs — used for structural connectivity profiling.

### Local Analysis Infrastructure

**`/wynton/group/abcd/CIAPM-analyses/`** (PI: xueyuan33)  
Active Python analysis project. Contains `abcd_utils/` data loading library, `code/` analysis scripts, and output figures showing ML model results (Lasso, Random Forest, XGBoost) predicting CBCL anxiety-depression and withdrawal-depression subscale scores from ABCD neuroimaging and behavioral data. Uses pyproject.toml with hatch build system.

**`/wynton/group/abcd/abcd-utils/`** (PI: pnedelec)  
Shared Python utility library (separate from CIAPM-analyses). Core dependencies: pandas, requests, matplotlib, tqdm, pyarrow, seaborn, scipy, pillow. Provides tools for loading and wrangling ABCD tabulated data.

---

## Part 3 — Project: Leukoaraiosis Detection Pipeline

### Scientific Goal

Build a deep learning model to detect **leukoaraiosis** (white matter hyperintensities/abnormalities) in the ABCD adolescent brain MRI dataset — as the foundational first step toward predicting longitudinal neurodevelopmental trajectories.

The full roadmap has two phases:
1. **Now (this project):** Single-timestamp detection of leukoaraiosis from T1w + T2w images
2. **Future:** Feed longitudinal pairs/triplets of multi-modal scans into a spatio-temporal model to predict the trajectory of white matter change over adolescent development

### The Core Challenges

**Challenge 1 — No FLAIR images**  
Clinical standard for white matter hyperintensity detection is FLAIR MRI, which suppresses CSF signal so lesions stand out brightly. ABCD only has T1w and T2w. On T2w, lesions and CSF both appear bright (hard to distinguish). On T1w, lesions appear dark (confused with gray matter). The model must learn the FLAIR-equivalent contrast implicitly by processing T1 and T2 together.

**Challenge 2 — No voxel-level labels**  
There are no manually segmented lesion masks in ABCD. Only image-level weak labels are available (WMH volume scores from automated pipelines, or clinical presence flags). This forces a **weakly supervised** approach: train a classifier on image-level labels, then extract spatial heatmaps from the model's internal activations to localize where it detected the pathology.

**Challenge 3 — Extreme class imbalance**  
In adolescents, leukoaraiosis is rare and subtle — typically <5% of white matter volume even when present. Standard accuracy and AUROC metrics are misleading under such imbalance. Must optimize for **AUPREC** (Area Under the Precision-Recall Curve), which focuses only on the rare positive class.

### The Technical Architecture

**Model:** Swin UNETR (Shifted Window Transformer U-Net)  
A 3D Vision Transformer where:
- Input: concatenated T1w + T2w → 2-channel 3D volume
- Encoder: hierarchical Swin Transformer (windowed self-attention; linear complexity)
- For Phase 1 (classification): CNN decoder bypassed; Global Average Pooling at bottleneck → MLP → sigmoid probability
- For Phase 5 (segmentation): decoder reactivated; trained on Grad-CAM pseudo-masks

**Loss function:** APLoss from LibAUC — a differentiable surrogate for AUPREC via squared hinge loss on pairwise rankings. Optimizer: SOAP (Stochastic Optimization of AP Curves).

**Heatmaps:** Grad-CAM on the final Swin Transformer block — backpropagates classification gradients to identify which 3D patches drove the positive prediction. Output thresholded and superimposed on T1w anatomy using nilearn.

### The Five-Phase Pipeline

| Phase | Goal | Status |
|-------|------|--------|
| 1.1 | Environment setup | **Complete** |
| 1.2 | Label extraction from ABCD data dictionary | Not started |
| 1.3 | MONAI preprocessing pipeline | Not started |
| 2 | Swin UNETR architectural modification | Not started |
| 3 | AUPREC training loop | Not started |
| 4 | Grad-CAM heatmap extraction | Not started |
| 5 | Pseudo-mask self-training | Not started |

---

## Part 4 — Phase 1.1: Environment Setup (Complete)

### What should be installed

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.11.13 | Runtime (system Python, used by venv) |
| **torch** | 2.5.1+cu124 | Core deep learning framework; cu124 = built against CUDA 12.4 |
| **torchvision** | 0.20.1+cu124 | Image utility ops (augmentation helpers) |
| **torchaudio** | 2.5.1+cu124 | Audio (bundled with torch; not used directly) |
| **monai** | 1.5.2 | Medical imaging pipeline: SwinUNETR, transforms, losses, data loading |
| **libauc** | 1.4.0 | Differentiable AUPREC: `APLoss` and `SOAP` optimizer |
| **nilearn** | 0.13.1 | Neuroimaging visualization: superimpose heatmaps on anatomy |
| **nibabel** | 5.4.2 | NIfTI (.nii.gz) file reading/writing |
| **einops** | 0.8.2 | Tensor reshaping (required by SwinUNETR internally) |
| **scikit-image** | 0.26.0 | Image processing utilities (MONAI dependency) |
| **scipy** | 1.17.1 | Scientific computing (MONAI + nilearn dependency) |
| **scikit-learn** | 1.8.0 | ML utilities (metrics, preprocessing) |
| **numpy** | 2.4.3 | Array operations |
| **pandas** | 3.0.2 | Tabular data handling (reading ABCD .tsv/.parquet) |
| **matplotlib** | 3.10.8 | Plotting |
| **pillow** | 12.1.1 | Image I/O |
| **tqdm** | 4.67.3 | Progress bars |
| **torch-geometric** | 2.7.0 | Graph neural networks (pulled in by LibAUC) |

### CUDA Is NOT Visible on the Login Node

`torch.cuda.is_available()` returns `False` on the login node because login nodes have no GPU. This is expected and correct. When a SLURM job runs on a GPU compute node (e.g., `--partition=gpu --gres=gpu:1`), the CUDA 12.5 module will be loaded and `torch.cuda.is_available()` will return `True`.

### SwinUNETR Instantiation Confirmed

```python
from monai.networks.nets import SwinUNETR
model = SwinUNETR(
    in_channels=2,        # T1w + T2w dual-channel input
    out_channels=2,       # background class + leukoaraiosis class
    feature_size=48,      # embedding dimension (standard config)
    use_checkpoint=True,  # gradient checkpointing = lower VRAM usage
    spatial_dims=3,       # 3D volumetric input
)
# → 62.2 million parameters
```

**Note on MONAI 1.5.2 API change:** In older versions of MONAI, SwinUNETR required `img_size=(96,96,96)` as a constructor argument. As of MONAI 1.5.2, this argument was removed — the model is now resolution-agnostic and the spatial dimensions are inferred at forward-pass time from the input tensor shape. The model is fully compatible with 96³, 128³, or any valid patch size.

### Verified Imports (All Pass)

```python
from monai.networks.nets import SwinUNETR          # model
from monai.losses import DiceFocalLoss             # Phase 5 loss
from monai.data import CacheDataset, ThreadDataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ConcatItemsd,
    Orientationd, Spacingd, NormalizeIntensityd,
    RandRotate90d, RandFlipd, RandShiftIntensityd, Compose
)
from libauc.losses import APLoss                   # AUPREC surrogate
from libauc.optimizers import SOAP                 # AUPREC optimizer
from nilearn import plotting, image                # visualization
```

The approach:
```python
import pandas as pd
dd = pd.read_csv("/wynton/group/abcd/6.0/datadictionary.csv")
# Search for WMH-related variables
mask = dd['description'].str.contains('white matter|hyperintensit|wmh|fazekas|leukoaraiosis',
                                       case=False, na=False)
dd[mask][['name','table_name','description','type_data']]
```

Once the label variable(s) are identified, the pipeline:
1. Loads the relevant tabulated table (parquet for speed)
2. Selects baseline session (`ses-00A`) rows
3. Binarizes: `label = (wmh_volume > threshold).astype(int)`
4. Matches participant IDs to available T1w/T2w NIfTI file paths under `mproc/`
5. Writes a `manifest.csv` with columns: `participant_id`, `t1w_path`, `t2w_path`, `label`

This manifest becomes the input to the MONAI data pipeline in Phase 1.3.
