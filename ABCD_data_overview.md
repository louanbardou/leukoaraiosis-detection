# ABCD Study — Data Overview
**Path:** `/wynton/group/abcd/`  
**Documented:** 2026-04-10  
**Data releases accessible:** 6.0, 6.1 (releases 4.0, 5.0, 5.1 exist but are permission-restricted)

---

## 1. What is the ABCD Study?

The **Adolescent Brain Cognitive Development (ABCD) Study** is the largest long-term study of brain development and child health in the United States. It tracks ~11,900 children from age 9–10 through early adulthood, collecting neuroimaging, cognitive assessments, mental and physical health measures, genetics, and rich socio-environmental data at regular intervals.

- **Total subjects in release 6.0:** 11,868 unique participants (`sub-XXXXXXXX` format)
- **Total rows (e.g. in demographics):** 67,410 (one per subject per session)
- **Sessions (time points):** 7 visits, ~annual
- **Study sites:** 21 research sites across the US

---

## 2. Study Timeline — Sessions

| Session ID  | Label                  | Approx. Age  | Notes                              |
|-------------|------------------------|--------------|------------------------------------|
| `ses-00A`   | Baseline               | 9–10 years   | Full protocol; initial enrollment  |
| `ses-01A`   | 1-Year Follow-Up       | 10–11 years  |                                    |
| `ses-02A`   | 2-Year Follow-Up       | 11–12 years  | Major imaging time point           |
| `ses-03A`   | 3-Year Follow-Up       | 12–13 years  |                                    |
| `ses-04A`   | 4-Year Follow-Up       | 13–14 years  |                                    |
| `ses-05A`   | 5-Year Follow-Up       | 14–15 years  |                                    |
| `ses-06A`   | 6-Year Follow-Up       | 15–16 years  |                                    |

---

## 3. Directory Structure

```
/wynton/group/abcd/
├── 4.0/                        # Earlier release — permission restricted
├── 5.0/                        # Earlier release — permission restricted
├── 5.1/                        # Earlier release — permission restricted
├── 6.0/                        # CURRENT main release
│   ├── datadictionary.csv      # Master data dictionary (92,419 entries)
│   ├── tabulated/              # All survey/behavioral/MRI summary data (718 files = 239 tables × 3 formats)
│   └── imaging/
│       └── derivatives/
│           ├── mproc/          # Per-subject processed imaging (808 subjects)
│           ├── mrtrix/         # Tractography — partial (626 subjects)
│           └── tractointersect/# Tract-ROI intersection matrices (626 subjects)
├── 6.1/                        # CURRENT incremental update
│   ├── tabulated/              # Updated tabulated data (same 239 tables)
│   └── concat/                 # Pre-concatenated imaging matrices (fast access)
│       ├── vertexwise/         # Vertex-level brain surface data (.mat)
│       └── voxelwise/          # Voxel-level diffusion data (.mat)
├── CIAPM-analyses/             # Local analysis code (Python; ML models: Lasso, RF, XGBoost)
├── abcd-utils/                 # Shared utility library (Python)
└── abcd-shared.code-workspace  # VS Code workspace file
```

---

## 4. Data Dictionary

**File:** `/wynton/group/abcd/6.0/datadictionary.csv`  
**Size:** 59.8 MB | **Rows:** 92,419 entries  
**Columns:** `id`, `domain`, `table_label`, `source`, `table_name`, `name`, `description`, `type_level`, `type_data`, `type_var`

This is the master reference for every variable in the release. Each row describes one variable: its table, human-readable description, data type (integer, float, string, categorical), and measurement level.

---

## 5. Tabulated Data — File Formats

Every table exists in three formats under `tabulated/`:

| Extension  | Format             | Use case                              |
|------------|--------------------|---------------------------------------|
| `.tsv`     | Tab-separated text | Human-readable, pandas-compatible     |
| `.parquet` | Columnar binary    | Fast I/O, smaller; preferred for code |
| `.json`    | Metadata sidecar   | Variable descriptions, levels, units  |

**ID columns present in every table:**
- `participant_id` — subject identifier, format `sub-XXXXXXXX`
- `session_id` — time point, format `ses-00A` through `ses-06A`

**Coding conventions (from JSON sidecars):**
- `n/a` — not applicable (item not administered at that time point)
- `555` — question not asked (version/wave issue)
- `888` — not yet assessed
- `777` / `999` — don't know / refused

---

## 6. Tabulated Data — Table Naming Convention

Tables follow the pattern: `{domain}_{respondent}_{instrument}`

**Respondent codes:**
| Code | Meaning                              |
|------|--------------------------------------|
| `p`  | Parent/caregiver report              |
| `y`  | Youth self-report                    |
| `l`  | Linked/geocoded (administrative data)|
| `t`  | Teacher report                       |
| `e`  | Examiner/rater                       |
| `g`  | General/administrative (study-level) |

---

## 7. Tabulated Data — Domains (239 tables total)

### 7.1 `ab` — ABCD Administrative (5 tables)

Core study metadata and demographics. Every participant has rows here.

| Table             | Description                                    | Variables |
|-------------------|------------------------------------------------|-----------|
| `ab_g_dyn`        | ABCD Dynamic Variables [General]               | 24        |
| `ab_g_stc`        | ABCD Static Variables [General]                | 49        |
| `ab_p_demo`       | Demographics [Parent]                          | 236       |
| `ab_p_ocp`        | Occupation Survey [Parent]                     | 56        |
| `ab_p_screen`     | Screener (Study Eligibility) [Parent]          | 90        |

**Key variables in `ab_p_demo`:** child age at collection, sex, race/ethnicity (multi-select), country of origin (child and both parents and grandparents), parental education, household income, insurance type, marital status, religiosity, language of administration, adoption status, household roster.

**Key variables in `ab_g_stc` / `ab_g_dyn`:** site ID, family ID, twin status, scanner model, study arm, ABCD version flags.

---

### 7.2 `covid` — COVID-19 (9 tables)

Collected during the pandemic period. Includes both self-report questionnaires and geocoded administrative data linked to participants' residential addresses.

| Table                | Description                                                                 | Variables |
|----------------------|-----------------------------------------------------------------------------|-----------|
| `covid_l_geoadm`     | COVID-19 Geocoded — Administrative Info                                     | 6         |
| `covid_l_geobls`     | COVID-19 Geocoded — Bureau of Labor Statistics (BLS)                        | 10        |
| `covid_l_geocen`     | COVID-19 Geocoded — Census Data                                             | 5         |
| `covid_l_geocount`   | COVID-19 Geocoded — JHU Covid-19 Prevalence (county-level)                 | 350       |
| `covid_l_geopolicy`  | COVID-19 Geocoded — CDC Policy Surveillance (state/county policy tracking)  | 711       |
| `covid_l_geosg`      | COVID-19 Geocoded — SafeGraph Social Distancing Metrics                     | 350       |
| `covid_p_qtn`        | COVID-19 Questionnaire [Parent]                                             | 295       |
| `covid_y_fitbqtn`    | COVID-19 Fitbit Post-Assessment Survey [Youth]                              | 38        |
| `covid_y_qtn`        | COVID-19 Questionnaire [Youth]                                              | 311       |

---

### 7.3 `ecb` — Endocannabinoid Substudy (1 table)

| Table       | Description                   | Variables |
|-------------|-------------------------------|-----------|
| `ecb_y_ecb` | Endocannabinoid Substudy [Youth] | 14     |

Substudy on endocannabinoid system — small subset of participants.

---

### 7.4 `fc` — Family & Culture (30 tables)

Covers parenting practices, family environment, peer relationships, school performance, acculturation, and neighborhood perceptions.

| Table           | Description                                                    | Variables |
|-----------------|----------------------------------------------------------------|-----------|
| `fc_p_aclt`     | Acculturation [Parent]                                         | 9         |
| `fc_p_drv`      | Youth Driving [Parent]                                         | 18        |
| `fc_p_fes`      | Family Environment Scale [Parent]                              | 69        |
| `fc_p_hsf`      | Home Short Form [Parent]                                       | 17        |
| `fc_p_meim`     | Multi-Group Ethnic Identity Measure [Parent]                   | 28        |
| `fc_p_nce`      | Neighborhood Collective Efficacy [Parent]                      | 19        |
| `fc_p_nsc`      | Neighborhood Safety & Crime [Parent]                           | 25        |
| `fc_p_pk`       | Parental Knowledge Scale [Parent]                              | 15        |
| `fc_p_psb`      | Prosocial Behavior [Parent]                                    | 8         |
| `fc_p_sag`      | School Attendance & Grades [Parent]                            | 26        |
| `fc_p_vs`       | Values Scale [Parent]                                          | 41        |
| `fc_y_aclt`     | Acculturation [Youth]                                          | 9         |
| `fc_y_as`       | Activity Space [Youth]                                         | 35        |
| `fc_y_crpbi`    | Children's Report of Parental Behavioral Inventory [Youth]     | 20        |
| `fc_y_eut`      | Experiences with Unfair Treatment [Youth]                      | 30        |
| `fc_y_fes`      | Family Environment Scale [Youth]                               | 25        |
| `fc_y_meim`     | Multi-Group Ethnic Identity Measure [Youth]                    | 28        |
| `fc_y_mnbs`     | Multidimensional Neglectful Behavior Scale [Youth]             | 17        |
| `fc_y_naa`      | Native American Acculturation [Youth]                          | 5         |
| `fc_y_nsc`      | Neighborhood Safety & Crime [Youth]                            | 22        |
| `fc_y_pbp`      | Peer Behavior Profile [Youth]                                  | 9         |
| `fc_y_pet`      | Pet Ownership [Youth]                                          | 10        |
| `fc_y_pm`       | Parental Monitoring [Youth]                                    | 10        |
| `fc_y_pnh`      | Peer Network Health [Youth]                                    | 11        |
| `fc_y_psb`      | Prosocial Behavior [Youth]                                     | 8         |
| `fc_y_rpi`      | Resistance to Peer Influence [Youth]                           | 15        |
| `fc_y_sag`      | School Attendance & Grades [Youth]                             | 5         |
| `fc_y_srpf`     | School Risk & Protective Factors [Youth]                       | 26        |
| `fc_y_vs`       | Values Scale [Youth]                                           | 41        |
| `fc_y_wpss`     | Wills Problem Solving Scale [Youth]                            | 11        |

---

### 7.5 `gn` — Genetics (3 tables)

| Table            | Description                       | Variables |
|------------------|-----------------------------------|-----------|
| `gn_e_zygrat`    | Twin Zygosity Rating [Examiner]   | 25        |
| `gn_y_genrel`    | Genetic Relatedness [Youth]       | 14        |
| `gn_y_popstruct` | Genetic Population Structure [Youth] | 32     |

`gn_y_genrel`: pairwise relatedness estimates (e.g. cryptic relatedness, twin pairs).  
`gn_y_popstruct`: principal components of ancestry (top PCs from genotyping array), used as covariates for genetic studies.

---

### 7.6 `irma` — Hurricane Irma Experiences (2 tables)

| Table        | Description                          | Variables |
|--------------|--------------------------------------|-----------|
| `irma_p_qtn` | Hurricane Irma Experiences [Parent]  | 128       |
| `irma_y_qtn` | Hurricane Irma Experiences [Youth]   | 114       |

Administered only to participants in affected regions. Covers exposure, displacement, stress, and impact of the 2017 hurricane.

---

### 7.7 `le` — Life Events & Linked Environment (69 tables)

The largest environmental/contextual domain. All `le_l_*` tables are **geocoded linked datasets** derived from participants' residential addresses — they are not self-report. Data comes from federal agencies (EPA, Census, CDC), academic research products, and commercial sources.

#### Administrative / Geographic Context
| Table             | Description                                              | Vars |
|-------------------|----------------------------------------------------------|------|
| `le_l_admin`      | Residential Address Description Variables                | 6    |
| `le_l_urban`      | Urban/Rural Area (Census)                                | 3    |
| `le_l_elevation`  | Elevation of Address (Google API)                        | 3    |

#### Socioeconomic & Deprivation
| Table             | Description                                              | Vars |
|-------------------|----------------------------------------------------------|------|
| `le_l_adi`        | Area Deprivation Index (ADI)                             | 57   |
| `le_l_coi`        | Child Opportunity Index 2.0 (COI)                        | 261  |
| `le_l_svi`        | Social Vulnerability Index (SVI) — CDC                   | 108  |
| `le_l_ssvi`       | Minority Health Social Vulnerability Index (MHSVI)       | 54   |
| `le_l_eci`        | Index of Concentration at the Extremes (ACS)             | 6    |
| `le_l_dissim`     | Dissimilarity Index — racial segregation (ACS)           | 12   |
| `le_l_entropy`    | Multi-group Entropy Index — diversity (ACS)              | 9    |
| `le_l_expint`     | Exposure/Interaction Index (ACS)                         | 12   |
| `le_l_rentmort`   | Rent and Mortgage Statistics (ACS)                       | 15   |
| `le_l_socmob`     | Social Mobility (Opportunity Atlas)                      | 21   |
| `le_l_nbhsoc`     | Neighborhood Socioeconomic Status (NaNDA)                | 24   |

#### Crime & Safety
| Table          | Description                          | Vars |
|----------------|--------------------------------------|------|
| `le_l_crime`   | Crime (ICPSR)                        | 24   |
| `le_l_gi`      | Getis-Ord Gi* Crime Hot Spots (ICPSR)| 24   |

#### Air Quality & Environmental Pollution
| Table              | Description                                                         | Vars |
|--------------------|---------------------------------------------------------------------|------|
| `le_l_no2`         | Satellite-based NO2 Measures                                        | 9    |
| `le_l_o3`          | Satellite-based O3 Measures                                         | 9    |
| `le_l_particulat`  | Satellite-based Particulate Measures (PM10, etc.)                   | 45   |
| `le_l_pm25`        | Satellite-based PM2.5 Measures                                      | 12   |
| `le_l_nata`        | Selected EJScreen Measures — environmental justice                  | 9    |
| `le_l_leadrisk`    | Lead Risk (Vox)                                                     | 9    |
| `le_l_prenatal`    | Pollution Measures for Prenatal Addresses                           | 4    |

#### Climate & Physical Environment
| Table           | Description                              | Vars |
|-----------------|------------------------------------------|------|
| `le_l_temp`     | Temperature Estimates (PRISM)            | 21   |
| `le_l_vpd`      | Vapor Pressure Deficit (PRISM)           | 21   |
| `le_l_noise`    | Environmental Noise Estimates (Harvard)  | 27   |
| `le_l_nlcd`     | Land Cover & Tree Canopy (NLCD)          | 90   |
| `le_l_urbsat`   | Land-use Measures (NLT)                  | 33   |
| `le_l_walk`     | Walkability (EPA)                        | 3    |
| `le_l_parks`    | Parks (NaNDA)                            | 12   |
| `le_l_roadprox` | Road Proximity (Kalibrate)               | 3    |
| `le_l_traffic`  | Traffic Density (Kalibrate)              | 3    |

#### Population Density
| Table             | Description                         | Vars |
|-------------------|-------------------------------------|------|
| `le_l_denspop`    | Population Density (EPA)            | 3    |
| `le_l_densbld`    | Building Density (EPA)              | 3    |
| `le_l_densveh`    | Vehicle Density (ACS)               | 6    |

#### Education & Opportunity
| Table                  | Description                                                     | Vars |
|------------------------|-----------------------------------------------------------------|------|
| `le_l_demo__cnty`      | School Demographics — County (SEDA)                            | 27   |
| `le_l_demo__distr`     | School Demographics — District (SEDA)                          | 29   |
| `le_l_demo__metro`     | School Demographics — Metro (SEDA)                             | 30   |
| `le_l_demo__schl`      | School Demographics — School (SEDA)                            | 26   |
| `le_l_math__cnty`      | School Math Achievement — County (SEDA)                        | 105  |
| `le_l_math__comz`      | School Math Achievement — Commuting Zone (SEDA)                | 105  |
| `le_l_math__distr`     | School Math Achievement — District (SEDA)                      | 105  |
| `le_l_math__metro`     | School Math Achievement — Metro (SEDA)                         | 105  |
| `le_l_read__cnty`      | School Reading Achievement — County (SEDA)                     | 105  |
| `le_l_read__comz`      | School Reading Achievement — Commuting Zone (SEDA)             | 105  |
| `le_l_read__distr`     | School Reading Achievement — District (SEDA)                   | 105  |
| `le_l_read__metro`     | School Reading Achievement — Metro (SEDA)                      | 105  |
| `le_l_pool__cnty`      | School Math & Reading Pool — County (SEDA)                     | 134  |
| `le_l_pool__comz`      | School Math & Reading Pool — Commuting Zone (SEDA)             | 134  |
| `le_l_pool__distr`     | School Math & Reading Pool — District (SEDA)                   | 134  |
| `le_l_pool__metro`     | School Math & Reading Pool — Metro (SEDA)                      | 134  |
| `le_l_pool__schl`      | School Math & Reading Pool — School (SEDA)                     | 20   |
| `le_l_artsports`       | Performing Arts & Sports Orgs (NaNDA)                          | 24   |
| `le_l_relciv`          | Religious/Civic Organizations (NaNDA)                          | 24   |
| `le_l_socsrv`          | Social Services (NaNDA)                                        | 12   |
| `le_l_aca`             | Affordable Care Act Medicaid Expansion Data (KFF)              | 3    |
| `le_l_oz`              | Opportunity Zones and Investment Scores (OZ)                   | 6    |

#### Substance Access / Policy
| Table              | Description                                              | Vars |
|--------------------|----------------------------------------------------------|------|
| `le_l_densalc`     | Alcohol Outlet Density (Census 2016)                     | 3    |
| `le_l_lawsmj`      | Cannabis Legalization Categories by State (NCSL/MPP)     | 1    |
| `le_l_medmj`       | Medical Marijuana Policy Data (OPTIC)                    | 18   |
| `le_l_goodsam`     | Good Samaritan Policy Data (OPTIC)                       | 6    |
| `le_l_polnalox`    | Naloxone Policy Data (OPTIC)                             | 9    |
| `le_l_rxmonit`     | Prescription Drug Monitoring Program Policy (OPTIC)      | 9    |
| `le_l_rxnalox`     | Co-prescribing Naloxone Policy Data (OPTIC)              | 6    |
| `le_l_rxopioid`    | CDC Opioid Prescription Dispensing per 100k (CDC)        | 18   |

#### Other
| Table             | Description                                             | Vars |
|-------------------|---------------------------------------------------------|------|
| `le_l_lodes`      | Number of Jobs and Job Density (LODES)                  | 54   |
| `le_l_attimm`     | Immigration Bias Measures (Hatzenbuehler)               | 1    |
| `le_l_censusret`  | Census Return Rate (Anomie/Social Capital)              | 6    |
| `le_l_places`     | Behavioral Health Measures (PLACES — CDC)               | 84   |

---

### 7.8 `mh` — Mental Health (58 tables)

The most comprehensive psychiatric domain. Covers parent-reported, youth self-reported, and clinician-administered measures. The KSADS (Schedule for Affective Disorders and Schizophrenia) modules provide structured diagnostic assessments.

#### Behavior Checklists & Ratings
| Table          | Description                                    | Variables |
|----------------|------------------------------------------------|-----------|
| `mh_p_abcl`    | Adult Behavior Checklist [Parent]              | 215       |
| `mh_p_asr`     | Adult Self Report [Parent]                     | 187       |
| `mh_p_cbcl`    | Child Behavior Checklist [Parent]              | 184       |
| `mh_p_ders`    | Difficulty in Emotion Regulation [Parent]      | 40        |
| `mh_p_eatq`    | Early Adolescent Temperament Questionnaire [Parent] | 91   |
| `mh_p_gbi`     | General Behavior Inventory — Mania [Parent]    | 15        |
| `mh_p_kbi`     | KSADS Background Items [Parent]                | 164       |
| `mh_p_ple`     | Life Events [Parent]                           | 222       |
| `mh_p_pss`     | Perceived Stress Scale [Parent]                | 13        |
| `mh_p_ssrs`    | Short Social Responsiveness Scale [Parent]     | 16        |
| `mh_p_famhx`   | Family Psychiatric History [Parent]            | 1,453     |
| `mh_t_bpm`     | Brief Problem Monitor [Teacher]                | 33        |
| `mh_y_bisbas`  | BIS/BAS Scales [Youth]                         | 36        |
| `mh_y_bpm`     | Brief Problem Monitor [Youth]                  | 34        |
| `mh_y_cb`      | Cyberbullying [Youth]                          | 16        |
| `mh_y_erq`     | Emotion Regulation Questionnaire [Youth]       | 13        |
| `mh_y_kbi`     | KSADS Background Items [Youth]                 | 62        |
| `mh_y_pai`     | NIH Toolbox Positive Affect Items [Youth]      | 14        |
| `mh_y_peq`     | Peer Experiences Questionnaire [Youth]         | 33        |
| `mh_y_ple`     | Life Events [Youth]                            | 218       |
| `mh_y_pps`     | Prodromal Psychosis Scale [Youth]              | 73        |
| `mh_y_resil`   | Resilience [Youth]                             | 11        |
| `mh_y_sup`     | 7-Up Mania Inventory [Youth]                   | 12        |
| `mh_y_upps`    | UPPS-P Impulsive Behavior Scale (Short) [Youth]| 33        |
| `mh_y_ysr`     | Youth Self Report [Youth]                      | 177       |

#### KSADS Diagnostic Modules (Parent-reported)
| Table                  | Disorder                                 | Variables |
|------------------------|------------------------------------------|-----------|
| `mh_p_ksads__adhd`     | ADHD                                     | 50        |
| `mh_p_ksads__agor`     | Agoraphobia                              | 14        |
| `mh_p_ksads__asd`      | Autism Spectrum Disorders                | 19        |
| `mh_p_ksads__bpd`      | Bipolar Disorders                        | 61        |
| `mh_p_ksads__cond`     | Conduct Disorder                         | 35        |
| `mh_p_ksads__dep`      | Depressive Disorders (MDD, PDD)          | 49        |
| `mh_p_ksads__dmdd`     | Disruptive Mood Dysregulation Disorder   | 5         |
| `mh_p_ksads__ed`       | Eating Disorders                         | 40        |
| `mh_p_ksads__gad`      | Generalized Anxiety Disorder             | 19        |
| `mh_p_ksads__hom`      | Homicidality                             | 6         |
| `mh_p_ksads__ocd`      | Obsessive-Compulsive Disorder            | 26        |
| `mh_p_ksads__odd`      | Oppositional Defiant Disorder            | 28        |
| `mh_p_ksads__panic`    | Panic Disorder                           | 18        |
| `mh_p_ksads__phobia`   | Specific Phobia                          | 13        |
| `mh_p_ksads__psych`    | Psychosis                                | 60        |
| `mh_p_ksads__ptsd`     | PTSD                                     | 60        |
| `mh_p_ksads__sepanx`   | Separation Anxiety                       | 29        |
| `mh_p_ksads__sleep`    | Sleep Problems                           | 4         |
| `mh_p_ksads__socanx`   | Social Anxiety Disorder                  | 19        |
| `mh_p_ksads__suic`     | Suicidality                              | 51        |
| `mh_p_ksads__tic`      | Tic Disorders                            | 17        |

#### KSADS Diagnostic Modules (Youth self-report — subset)
| Table                  | Disorder                                 | Variables |
|------------------------|------------------------------------------|-----------|
| `mh_y_ksads__bpd`      | Bipolar Disorders                        | 61        |
| `mh_y_ksads__cond`     | Conduct Disorder                         | 35        |
| `mh_y_ksads__dep`      | Depressive Disorders                     | 49        |
| `mh_y_ksads__dmdd`     | Disruptive Mood Dysregulation Disorder   | 5         |
| `mh_y_ksads__ed`       | Eating Disorders                         | 40        |
| `mh_y_ksads__gad`      | Generalized Anxiety Disorder             | 19        |
| `mh_y_ksads__ocd`      | OCD                                      | 26        |
| `mh_y_ksads__panic`    | Panic Disorder                           | 18        |
| `mh_y_ksads__ptsd`     | PTSD                                     | 60        |
| `mh_y_ksads__sleep`    | Sleep Problems                           | 4         |
| `mh_y_ksads__socanx`   | Social Anxiety Disorder                  | 19        |
| `mh_y_ksads__suic`     | Suicidality                              | 51        |

KSADS variables encode: past symptom (`__past_sx`), present symptom (`__pres_sx`), past diagnosis (`__past_dx`), present diagnosis (`__pres_dx`), partial remission (`__partrem_dx`).

---

### 7.9 `mr` — MRI (420 tables)

The largest domain by table count. All respondent-coded `y` (youth). Covers structural MRI, diffusion MRI, resting-state fMRI, three task-based fMRI paradigms, and MRI quality control.

#### MRI Administration
| Table              | Description                          | Variables |
|--------------------|--------------------------------------|-----------|
| `mr_y_adm__info`   | MRI Info (scanner, site)             | 7         |
| `mr_y_adm__nts`    | Scanning Checklist and Notes         | 21        |
| `mr_y_adm__qtn`    | Pre/Post-Scan Questionnaires         | 48        |

#### MRI Quality Control
| Table                      | Description                                        | Variables |
|----------------------------|----------------------------------------------------|-----------|
| `mr_y_qc__clfind`          | Clinical Findings QC                               | varies    |
| `mr_y_qc__incl`            | Inclusion/exclusion QC flags                       | varies    |
| `mr_y_qc__mot`             | Motion QC metrics                                  | varies    |
| `mr_y_qc__post__aut`       | Post-processing automated QC                       | varies    |
| `mr_y_qc__post__man__dmri` | Post-processing manual QC — dMRI                   | varies    |
| `mr_y_qc__post__man__fmri` | Post-processing manual QC — fMRI                   | varies    |
| `mr_y_qc__post__man__fsurf`| Post-processing manual QC — FreeSurfer surface     | varies    |
| `mr_y_qc__post__man__t2`   | Post-processing manual QC — T2w                    | varies    |
| `mr_y_qc__raw__dmri`       | Raw dMRI QC                                        | varies    |
| `mr_y_qc__raw__event`      | Raw event-related fMRI QC                          | varies    |
| `mr_y_qc__raw__rsfmri`     | Raw resting-state fMRI QC                          | varies    |
| `mr_y_qc__raw__smri__t1`   | Raw sMRI T1w QC                                    | varies    |
| `mr_y_qc__raw__smri__t2`   | Raw sMRI T2w QC                                    | varies    |
| `mr_y_qc__raw__tfmri__all` | Raw task fMRI QC — all tasks                       | varies    |
| `mr_y_qc__raw__tfmri__mid` | Raw task fMRI QC — MID task                        | varies    |
| `mr_y_qc__raw__tfmri__nback`| Raw task fMRI QC — N-Back task                    | varies    |
| `mr_y_qc__raw__tfmri__sst` | Raw task fMRI QC — Stop Signal task                | varies    |

#### Structural MRI (sMRI) — Tabulated ROI Summaries
Uses FreeSurfer parcellations. Atlases: **Desikan (dsk)**, **Destrieux (dst)**, **Fuzzy (fzy)**, subcortical **ASEG**.

| Table                         | Measure                                  | Atlas         | Variables |
|-------------------------------|------------------------------------------|---------------|-----------|
| `mr_y_smri__vol__aseg`        | Subcortical volume                       | ASEG          | ~30       |
| `mr_y_smri__vol__dsk`         | Cortical volume                          | Desikan       | 71        |
| `mr_y_smri__vol__dst`         | Cortical volume                          | Destrieux     | 151       |
| `mr_y_smri__vol__fzy`         | Cortical volume                          | Fuzzy         | varies    |
| `mr_y_smri__thk__dsk`         | Cortical thickness                       | Desikan       | 71        |
| `mr_y_smri__thk__dst`         | Cortical thickness                       | Destrieux     | 151       |
| `mr_y_smri__thk__fzy`         | Cortical thickness                       | Fuzzy         | varies    |
| `mr_y_smri__area__dsk`        | Cortical surface area                    | Desikan       | 71        |
| `mr_y_smri__area__dst`        | Cortical surface area                    | Destrieux     | 151       |
| `mr_y_smri__area__fzy`        | Cortical surface area                    | Fuzzy         | varies    |
| `mr_y_smri__sulc__dsk`        | Sulcal depth                             | Desikan       | 71        |
| `mr_y_smri__sulc__dst`        | Sulcal depth                             | Destrieux     | 151       |
| `mr_y_smri__sulc__fzy`        | Sulcal depth                             | Fuzzy         | varies    |
| `mr_y_smri__t1__aseg`         | T1w intensity (subcortical)              | ASEG          | varies    |
| `mr_y_smri__t1__gm__dsk/dst/fzy` | T1w intensity (gray matter)           | multiple      | varies    |
| `mr_y_smri__t1__gwc__dsk/dst/fzy`| T1w gray/white contrast               | multiple      | varies    |
| `mr_y_smri__t1__wm__dsk/dst/fzy` | T1w intensity (white matter)          | multiple      | varies    |
| `mr_y_smri__t2__*`            | T2w intensity (same breakdown as T1w)    | multiple      | varies    |

#### Diffusion MRI — DTI (Full shell and Inner shell)
Two acquisition variants: **Full shell (fs)** and **Inner shell (is)**. Metrics: FA (Fractional Anisotropy), LD (Longitudinal/Axial Diffusivity), MD (Mean Diffusivity), TD (Transverse/Radial Diffusivity). Parcellations: ASEG, AtlasTrack (at), Gray Matter (gm), Gray/White Contrast (gwc), White Matter (wm) × Desikan (dsk) / Destrieux (dst).

Pattern: `mr_y_dti__{fs|is}__{fa|ld|md|td}__{aseg|at|gm__dsk|gm__dst|gwc__dsk|gwc__dst|wm__dsk|wm__dst}`  
Also: `mr_y_dti__{fs|is}__vol__at` (AtlasTrack tract volumes)

Total DTI tables: ~72 (36 per shell × 2 shells)

#### Diffusion MRI — RSI (Restriction Spectrum Imaging)
RSI provides multi-compartment diffusion measures. Metrics per hemisphere/tissue:
- `fni` = free normalized isotropic
- `hnd/hni/hnt` = hindered normalized directional/isotropic/total
- `rnd/rni/rnt` = restricted normalized directional/isotropic/total

Pattern: `mr_y_rsi__{fni|hnd|hni|hnt|rnd|rni|rnt}__{aseg|at|gm__dsk|gm__dst|gwc__dsk|gwc__dst|wm__dsk|wm__dst}`

Total RSI tables: ~56

#### Resting-State fMRI (rsfMRI)
| Table                        | Description                                         | Variables |
|------------------------------|-----------------------------------------------------|-----------|
| `mr_y_rsfmri__corr__gpnet`   | Gordon network parcel correlations                  | varies    |
| `mr_y_rsfmri__corr__gpnet__aseg` | Gordon network × ASEG correlations              | varies    |
| `mr_y_rsfmri__var__aseg`     | BOLD variance — subcortical (ASEG)                  | varies    |
| `mr_y_rsfmri__var__dsk`      | BOLD variance — cortical (Desikan)                  | varies    |
| `mr_y_rsfmri__var__dst`      | BOLD variance — cortical (Destrieux)                | varies    |
| `mr_y_rsfmri__var__gp`       | BOLD variance — Gordon parcels                      | varies    |

#### Task fMRI — MID (Monetary Incentive Delay)
Reward anticipation/receipt paradigm. Contrasts: anticipation (large/small/neutral reward/loss) vs. baseline, many combinations. ROI parcellations: ASEG, Desikan (dsk), Destrieux (dst). Two runs (r01, r02) available separately plus combined.

Behavioral data: `mr_y_tfmri__mid__beh`, `mr_y_tfmri__mid__qtn`

Example tables: `mr_y_tfmri__mid__alvn__aseg/dsk/dst` (anticipation large vs. neutral), `mr_y_tfmri__mid__rpvnf__aseg/dsk/dst` (reward positive vs. no feedback), etc. ~60 MID contrast tables.

#### Task fMRI — N-Back (Working Memory / Emotion)
2-back vs. 0-back working memory; emotional face recognition (positive/negative). Contrasts: 2b, 0b, 2bv0b, emo, emovntf, fvplc, ngfvntf, psfvntf, plc. Also recognition memory: `mr_y_tfmri__nback__rec`.

Behavioral data: `mr_y_tfmri__nback__beh`

~70+ N-Back contrast tables across ASEG, Desikan, Destrieux parcellations × 2 runs.

#### Task fMRI — SST (Stop Signal Task)
Response inhibition. Contrasts: correct stop vs. correct go (csvcg), incorrect stop vs. correct go (isvcg), any stop vs. correct go (asvcg), correct go vs. fixation (cgvfx), incorrect go vs. incorrect stop (igvcg/igvis), etc.

Behavioral data: `mr_y_tfmri__sst__beh`

~60+ SST contrast tables.

#### MR Spectroscopy (MRS)
| Table          | Description                          | Variables |
|----------------|--------------------------------------|-----------|
| `mrs_y_2dj`    | MRS — 2D J-resolved spectroscopy     | varies    |
| `mrs_y_hermes` | MRS — HERMES GABA/Glx editing        | varies    |

---

### 7.10 `nc` — Neurocognition (15 tables)

All youth-administered cognitive and behavioral tasks.

| Table          | Description                                                    | Variables |
|----------------|----------------------------------------------------------------|-----------|
| `nc_p_bdefs`   | Barkley Deficits in Executive Functioning Scale [Parent]       | 26        |
| `nc_y_bird`    | Behavioral Indicator of Resiliency to Distress Task [Youth]    | 34        |
| `nc_y_cct`     | Cash Choice Task (delay discounting, short) [Youth]            | 3         |
| `nc_y_ddis`    | Delay Discounting [Youth]                                      | 36        |
| `nc_y_ehis`    | Edinburgh Handedness Inventory (Short Form) [Youth]            | 8         |
| `nc_y_est`     | Emotional Stroop Task [Youth]                                  | 57        |
| `nc_y_flnkr`   | Flanker Task [Youth]                                           | 29        |
| `nc_y_gdt`     | Game of Dice (risk-taking) [Youth]                             | 26        |
| `nc_y_lmt`     | Little Man Task (mental rotation) [Youth]                      | 65        |
| `nc_y_nihtb`   | NIH Toolbox [Youth]                                            | 163       |
| `nc_y_ravlt`   | Rey Auditory Verbal Learning Test [Youth]                      | 35        |
| `nc_y_sit`     | Social Influence Task [Youth]                                  | 26        |
| `nc_y_smarte`  | Stanford Mental Arithmetic Response Time Evaluation [Youth]    | 66        |
| `nc_y_svs`     | Snellen Vision Screener [Youth]                                | 6         |
| `nc_y_wisc`    | WISC-V Matrix Reasoning [Youth]                                | 36        |

**NIH Toolbox (`nc_y_nihtb`, 163 vars):** Covers multiple cognitive domains — Fluid Composite, Crystallized Composite, and individual tests (Picture Vocabulary, Oral Reading Recognition, Pattern Comparison Processing Speed, List Sorting Working Memory, Dimensional Change Card Sort, Flanker Inhibitory Control, Picture Sequence Memory). Scores: uncorrected, age-corrected, and fully-corrected normed standard scores.

---

### 7.11 `nt` — Technology & Screen Time (15 tables)

| Table           | Description                                             | Variables |
|-----------------|---------------------------------------------------------|-----------|
| `nt_p_earsp`    | EARS Pre/Post Survey Pilot [Parent]                     | 95        |
| `nt_p_earsq`    | EARS Post-Assessment Survey [Parent]                    | 35        |
| `nt_p_fitbp`    | Fitbit Pre/Post Survey Pilot [Parent]                   | 77        |
| `nt_p_fitbq`    | Fitbit Pre/Post Survey [Parent]                         | 54        |
| `nt_p_stq`      | Parent Screen Time Questionnaire [Parent]               | 49        |
| `nt_p_yst`      | Screen Time Questionnaire [Parent]                      | 52        |
| `nt_y_earsadm`  | EARS Administrative Information [Youth]                 | 7         |
| `nt_y_earsapp`  | EARS Device Usage — App Usage Statistics [Youth]        | 319       |
| `nt_y_earskey`  | EARS Device Usage — Keyboard Input Statistics [Youth]   | 1,375     |
| `nt_y_earsp`    | EARS Pre/Post Survey Pilot [Youth]                      | 53        |
| `nt_y_earsplt`  | EARS Device Usage Statistics Pilot [Youth]              | 373       |
| `nt_y_earsq`    | EARS Post-Assessment Survey [Youth]                     | 24        |
| `nt_y_fitbp`    | Fitbit Pre/Post Survey Pilot [Youth]                    | 53        |
| `nt_y_fitbq`    | Fitbit Pre/Post Survey [Youth]                          | 23        |
| `nt_y_stq`      | Screen Time Questionnaire [Youth]                       | 181       |

**EARS** = Ecological momentary assessment of digital media use (passive phone sensing). Captures objective app usage patterns (319 app categories), keyboard input patterns (1,375 variables), and subjective experience surveys. Fitbit data covers physical activity and sleep.

---

### 7.12 `ph` — Physical Health (28 tables)

| Table          | Description                                                      | Variables |
|----------------|------------------------------------------------------------------|-----------|
| `ph_p_anthr`   | Anthropometrics [Parent]                                         | 13        |
| `ph_p_bfq`     | Breast Feeding Questionnaire [Parent]                            | 84        |
| `ph_p_bkfs`    | Block Kids Food Screener [Parent]                                | 209       |
| `ph_p_cna`     | Child Nutrition Assessment [Parent]                              | 21        |
| `ph_p_covid`   | Annual COVID Survey [Parent]                                     | 155       |
| `ph_p_dhx`     | Developmental History [Parent]                                   | 242       |
| `ph_p_ipaq`    | International Physical Activity Questionnaire [Parent]           | 18        |
| `ph_p_meds`    | Medications Inventory [Parent]                                   | 393       |
| `ph_p_mhx`     | Medical History [Parent]                                         | 144       |
| `ph_p_otbi`    | Ohio State TBI Screen — Baseline [Parent]                        | 85        |
| `ph_p_pds`     | Pubertal Development Scale & Menstrual History [Parent]          | 28        |
| `ph_p_saiq`    | Sports and Activities Involvement Questionnaire [Parent]         | 836       |
| `ph_p_sds`     | Sleep Disturbance Scale for Children [Parent]                    | 29        |
| `ph_p_sex`     | Sexual Orientation & Communication [Parent]                      | 9         |
| `ph_y_anthr`   | Anthropometrics [Youth]                                          | 14        |
| `ph_y_bkfs`    | Block Kids Food Screener [Youth]                                 | 211       |
| `ph_y_bld`     | Blood Draw [Youth]                                               | 56        |
| `ph_y_bp`      | Blood Pressure [Youth]                                           | 29        |
| `ph_y_covid`   | Annual COVID Survey [Youth]                                      | 48        |
| `ph_y_mctq`    | Munich Chronotype Questionnaire [Youth]                          | 119       |
| `ph_y_meds`    | Medications Inventory [Youth]                                    | 213       |
| `ph_y_pa`      | Youth Risk Behavior Survey — Physical Activity [Youth]           | 6         |
| `ph_y_pds`     | Pubertal Development Scale & Menstrual History [Youth]           | 39        |
| `ph_y_phs`     | Pubertal Hormone Saliva Analysis [Youth]                         | 47        |
| `ph_y_pq`      | Pain Questionnaire [Youth]                                       | 83        |
| `ph_y_rq`      | Respiratory Functioning [Youth]                                  | 27        |
| `ph_y_saiq`    | Sports & Activities — Reading & Music [Youth]                    | 239       |
| `ph_y_sex`     | Sexual Behavior, Orientation & Communication [Youth]             | 74        |

**Notable:** `ph_p_meds` (393 vars) and `ph_y_meds` (213 vars) capture all medications by class; `ph_y_bld` (56 vars) includes CBC and metabolic panel from the blood draw; `ph_y_phs` has salivary hormone levels (testosterone, DHEA, estradiol); `ph_p_dhx` (242 vars) captures full prenatal, perinatal, and early developmental history.

---

### 7.13 `sdev` — Social Development (15 tables)

| Table          | Description                                          | Variables |
|----------------|------------------------------------------------------|-----------|
| `sdev_p_apq`   | Alabama Parenting Questionnaire [Parent]             | 86        |
| `sdev_p_ders`  | Difficulties in Emotion Regulation [Parent]          | 74        |
| `sdev_p_fa`    | Firearms (BRFSS) [Parent]                            | 8         |
| `sdev_p_nbh`   | Perception of Neighborhood Scale [Parent]            | 52        |
| `sdev_p_pd`    | Personality Disposition [Parent]                     | 102       |
| `sdev_p_rd`    | Reported Delinquency [Parent]                        | 202       |
| `sdev_p_vict`  | Victimization [Parent]                               | 1,681     |
| `sdev_p_vt`    | Visit Type [Parent]                                  | 15        |
| `sdev_y_apq`   | Alabama Parenting Questionnaire [Youth]              | 110       |
| `sdev_y_ders`  | Difficulties in Emotion Regulation [Youth]           | 74        |
| `sdev_y_fa`    | Firearms (YRBSS) [Youth]                             | 9         |
| `sdev_y_pb`    | Peer Behavior [Youth]                                | 36        |
| `sdev_y_pd`    | Personality Disposition [Youth]                      | 242       |
| `sdev_y_rd`    | Reported Delinquency [Youth]                         | 356       |
| `sdev_y_vict`  | Victimization [Youth]                                | 1,675     |

**Note:** `sdev_p_vict` (1,681 vars) and `sdev_y_vict` (1,675 vars) are the two largest tables in the dataset, encoding detailed victimization history across many event types.

---

### 7.14 `su` — Substance Use (46 tables)

Covers alcohol, cannabis, tobacco/nicotine, vaping, and other drugs. Includes both self-report questionnaires and biological toxicology assays.

#### Biological Toxicology (objective measures)
| Table            | Description                            | Variables |
|------------------|----------------------------------------|-----------|
| `su_y_alctox`    | Alcohol Toxicology (blood/urine)       | 36        |
| `su_y_hairtox`   | Hair Drug Toxicology                   | 119       |
| `su_y_nictox`    | Nicotine Toxicology                    | 30        |
| `su_y_oftox`     | Oral Fluid (saliva) Toxicology         | 81        |
| `su_y_udstox`    | Urine Drug Toxicology                  | 112       |

#### KSADS Diagnostic — Substance Use Disorders
| Table              | Description                         | Variables |
|--------------------|-------------------------------------|-----------|
| `su_p_ksads__aud`  | Alcohol Use Disorder [Parent]       | 36        |
| `su_p_ksads__dud`  | Drug Use Disorders [Parent]         | 359       |
| `su_y_ksads__aud`  | Alcohol Use Disorder [Youth]        | 36        |
| `su_y_ksads__dud`  | Drug Use Disorders [Youth]          | 359       |

#### Alcohol
| Table           | Description                                   | Variables |
|-----------------|-----------------------------------------------|-----------|
| `su_y_alcexp`   | Alcohol Expectancies (AEQ-AB)                 | 14        |
| `su_y_alchss`   | Alcohol Hangover Symptom Scale (HSS)          | 34        |
| `su_y_alcmot`   | Alcohol Motives                               | 23        |
| `su_y_alcprob`  | Alcohol Problem Index (RAPI)                  | 41        |
| `su_y_alcsre`   | Alcohol Subjective Response and Effects (SRE) | 28        |

#### Cannabis / Marijuana
| Table         | Description                                | Variables |
|---------------|--------------------------------------------|-----------|
| `su_y_mjexp`  | Marijuana Expectancies (MEEQ-B)            | 13        |
| `su_y_mjmot`  | Marijuana Motives                          | 28        |
| `su_y_mjprob` | Marijuana Problem Index (MAPI)             | 23        |
| `su_y_mjsre`  | Marijuana Subjective Response and Effects  | 20        |
| `su_y_mjws`   | Marijuana Withdrawal (CWS)                 | 20        |
| `su_y_mjexp`  | Marijuana Expectancies                     | 13        |

#### Nicotine / Tobacco / Vaping
| Table              | Description                               | Variables |
|--------------------|-------------------------------------------|-----------|
| `su_y_cigexp`      | Cigarette Expectancies (ASCQ)             | 16        |
| `su_y_cigmot`      | Tobacco Motives                           | 22        |
| `su_y_nicdpnd`     | PATH Nicotine Dependence                  | 13        |
| `su_y_nicsre`      | Nicotine Subjective Response and Effects  | 21        |
| `su_y_nicvapeexp`  | ENDS Expectancies (CEQ)                   | 15        |
| `su_y_nicvapereas` | Reasons for ENDS Use                      | 27        |
| `su_y_vapeexp`     | Vaping Expectancies                       | 12        |
| `su_y_vapemot`     | Vaping Motives                            | 12        |

#### Use Frequency & General Instruments
| Table           | Description                                       | Variables |
|-----------------|---------------------------------------------------|-----------|
| `su_y_sui`      | Substance Use Interview                           | 836       |
| `su_y_tlfb`     | Timeline Followback Interview Results             | 883       |
| `su_y_mysu`     | Mid-Year Substance Use Phone Interview            | 133       |
| `su_y_mypi`     | Mid-Year Phone Interview Introduction             | 30        |
| `su_y_lowuse`   | Low Level Use Questionnaire                       | 51        |
| `su_y_caff`     | Caffeine Use Questionnaire                        | 90        |
| `su_y_drgprob`  | Drug Problem Index (DAPI)                         | 23        |
| `su_y_otu`      | Opportunity to Use                                | 13        |
| `su_y_ptu`      | Peer Tolerance SU                                 | 19        |
| `su_y_perc`     | Perceived Harm SU                                 | 19        |
| `su_y_pgd`      | Peer Deviance SU                                  | 13        |
| `su_y_sibu`     | Sibling Use                                       | 33        |
| `su_y_itu`      | PATH Intention to Use                             | 18        |
| `su_p_des`      | Substance Use Density, Storage, and Exposure [Parent] | 381   |
| `su_p_rule`     | Parental Rules on Substance Use [Parent]          | 16        |
| `su_p_crpf`     | Community Risk and Protective Factors [Parent]    | 16        |
| `su_y_crpf`     | Community Risk and Protective Factors [Youth]     | 21        |
| `su_p_plus`     | Participant Last Use Survey [Parent]              | 103       |
| `su_y_plus`     | Participant Last Use Survey [Youth]               | 148       |

---

## 8. Imaging Data — Raw and Derived

### 8.1 Per-Subject Derivatives (`/wynton/group/abcd/6.0/imaging/derivatives/`)

#### `mproc/` — Minimally Processed Imaging
- **808 subjects** with at least one session processed
- Sessions available per subject: `ses-00A` (and later time points)
- **Structure per subject:** `sub-XXXXXXXX/ses-XXXX/{anat, dwi}/`
  - `anat/` — structural MRI outputs (T1w, T2w; FreeSurfer-derived surfaces and volumes)
  - `dwi/` — diffusion MRI outputs (FA maps, MD maps, tractography-ready data)

#### `mrtrix/` — MRtrix Tractography (partial)
- **3 subjects** (early/pilot; very limited)
- Per subject: `ses-00A/` with subfolders `bnst_l`, `bnst_r`, `scc_l`, `scc_r`, `registration_to_MNI`
- Contains tract-specific files for bed nucleus of the stria terminalis (BNST) and subgenual cingulate cortex (SCC) — bilateral

#### `tractointersect/` — Tract-ROI Intersection Matrices
- **626 subjects** × sessions `ses-00A`, `ses-02A`, `ses-04A`, `ses-06A` (even-numbered visits)
- Per subject/session: tract × ROI intersection matrices used for white matter profiling

---

### 8.2 Concatenated Imaging Matrices (`/wynton/group/abcd/6.1/concat/`)

Pre-assembled `.mat` files for fast loading — one matrix per measure covering all subjects. Organized by imaging modality and spatial resolution.

#### `vertexwise/` — Cortical Surface Data (vertex-level)

| Subfolder  | Modality | Measures                            | Smoothing kernels            | Hemispheres |
|------------|----------|-------------------------------------|------------------------------|-------------|
| `dti/`     | DTI      | `fa` (FA), `ld` (long. diffusivity) | sm0, sm16, sm256, sm1000     | lh, rh      |
| `rsi/`     | RSI      | `fni`, `hnt` (+ others)             | sm0, sm16, sm256, sm1000     | lh, rh      |
| `smri/`    | sMRI     | `area`, `thk`, `vol`                | sm0, sm16, sm256, sm1000     | lh, rh      |

File naming: `{measure}_{tissue}_{smoothing}_{hemisphere}.mat`  
e.g. `fa_gm_sm16_lh.mat` — FA in gray matter, 16 mm smooth, left hemisphere  
Also: `vol_info.mat` — subject/session index for the matrices

#### `voxelwise/` — Volumetric Diffusion Data (voxel-level)

| Subfolder | Modality | Measures                                                                                          |
|-----------|----------|---------------------------------------------------------------------------------------------------|
| `dti/`    | DTI      | `fa.mat`, `md.mat`                                                                                |
| `rsi/`    | RSI      | `fi`, `fni`, `hdf`, `hd`, `hif`, `hi`, `hnd`, `hni`, `hnt`, `ht`, `rd`, `rdf`, `rif`, `ri`, `rnd`, `rni`, `rnt`, `rt` |

Also: `vol_info.mat` in each subfolder for subject/session labels.

**RSI voxelwise metrics key:**
- `fi/fni` = free (normalized) isotropic
- `hd/hnd/hni/hnt/hdf/hif` = hindered (directional/normalized/isotropic/total/free)
- `rd/rnd/rni/rnt/rdf/rif` = restricted (directional/normalized/isotropic/total/free)

---

## 9. Summary Statistics

| Metric                         | Value                              |
|--------------------------------|------------------------------------|
| Total subjects                 | 11,868                             |
| Study sessions (time points)   | 7 (`ses-00A` through `ses-06A`)    |
| Total tabulated tables         | 239                                |
| Total tabulated files          | 718 (239 × TSV + Parquet + JSON)   |
| Data dictionary entries        | 92,419 variables                   |
| Imaging subjects (mproc)       | 808                                |
| Imaging subjects (tractointersect) | 626                            |
| Accessible releases            | 6.0 (full), 6.1 (updated tabulated + concat) |

---

## 10. Local Analysis Infrastructure

### `CIAPM-analyses/` (PI: xueyuan33)
A Python project for ABCD-based predictive modeling. Contains:
- `abcd_utils/` — data loading utilities
- `code/` — analysis scripts
- Pre-generated figures: Lasso, Random Forest, XGBoost model outputs (e.g. predicting CBCL anxdep, withdep scores)
- `pyproject.toml` — Python package definition

### `abcd-utils/` (PI: pnedelec)
A shared Python utility library for ABCD data access at this site.
- `abcd_utils/` — core module
- Versioned separately from CIAPM-analyses but shares the same library structure

### `.venv/` 
Shared Python virtual environment for the group.
