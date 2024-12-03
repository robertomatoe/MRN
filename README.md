# MRN Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Study Design](#study-design)
3. [Data Collection Methods](#data-collection-methods)
4. [Dataset Components](#dataset-components)
   - [Participant Demographics](#participant-demographics)
   - [Neuroimaging Data](#neuroimaging-data)
   - [Cognitive Assessments](#cognitive-assessments)
   - [MRN Metrics](#mrn-metrics)
   - [Clinical Outcomes](#clinical-outcomes)
5. [Statistical Methods](#statistical-methods)
6. [Quality Control](#quality-control)
7. [Technical Details](#technical-details)
8. [Appendices](#appendices)

---

## Project Overview

### Purpose
The Metacognitive Resilience Network (MRN) project aims to investigate the preservation of metacognitive abilities in early-stage Alzheimer's Disease (AD). The study hypothesizes the existence of a neural network, the MRN, that operates independently of structural brain deterioration, offering potential for targeted interventions.

### Objectives
1. Identify the components of the Metacognitive Resilience Network.
2. Analyze its role in cognitive resilience in AD.
3. Leverage multimodal neuroimaging and cognitive data to inform clinical outcomes.

### Key Features
- **Multimodal Integration**: Combines high-resolution neuroimaging, cognitive assessments, and clinical outcomes.
- **Automated Pipeline**: Ensures reproducibility and efficiency in data processing.
- **Clinical Relevance**: Tracks disease progression and evaluates intervention strategies.

---

## Study Design

### Population Characteristics

#### Inclusion Criteria
- **Age**: 40–100 years.
- **Cognitive Status**: Normal to moderate Alzheimer’s Disease (AD).
- **Consent**: Full capacity required to provide informed consent.

#### Exclusion Criteria
- **Severe Cognitive Impairment**: Participants with severe AD or other forms of advanced cognitive decline.
- **Non-AD Neurological Conditions**: Exclusion of participants with conditions such as Parkinson’s Disease or stroke-related cognitive impairment.
- **MRI Contraindications**: Includes implants, severe claustrophobia, or inability to remain still during scans.

---

### Sampling Framework

- **Method**: Recruitment from community and clinic-based sources to ensure diverse representation.
- **Location**: Recruitment sites focused on the US Midwest region.
- **Statistical Power**:
  - Target Power: 95%.
  - Effect Size: \(d = 0.3\).
  - Significance Level: \(\alpha = 0.05\).

---

### Study Timeline

#### Milestones
1. **Participant Recruitment**: Months 1–6.
2. **Baseline Data Collection**: Months 2–8.
3. **Follow-Up Visits**: Semi-annual follow-ups for 2 years.
4. **Final Analysis**: Months 30–36.

#### Data Collection Windows
- **Baseline**: Initial assessments conducted within 14 days of recruitment.
- **Follow-Up**: Assessments conducted every 180 days, allowing a ±14-day window for scheduling flexibility.

---

## Data Collection Methods

### Overview
The MRN project uses a multimodal approach to collect demographic, neuroimaging, cognitive, and clinical data. Standardized protocols ensure data consistency and quality across all participants.

---

### Demographics Module

#### Core Variables
1. **Participant ID**  
   - **Format**: `MRNxxx` (3-digit unique identifier).  
   - **Error Rate**: <0.001%.  
   - **Validation**: Checksum-based verification.  

2. **Age**  
   - **Resolution**: 0.01 years.  
   - **Range**: 42.5–95.6 years.  
   - **Distribution**: Normal (\(W = 0.994\)).  

3. **Sex**  
   - Categories: Male, Female, or Unknown.  

4. **Education**  
   - **Measure**: Completed academic years.  
   - **Categories**:
     - Low: <12 years (15.3%).  
     - Medium: 12–16 years (52.7%).  
     - High: >16 years (32.0%).  
   - **Reliability**: \( \text{ICC} = 0.97 \).  

5. **APOE Status**  
   - **Method**: PCR + HhaI restriction assay.  
   - **Quality Control**: 5% duplicate testing.  
   - **Distribution**:
     - \( \varepsilon 33 \): 24.1%.  
     - \( \varepsilon 34 \): 16.6%.  
     - \( \varepsilon 23 \): 5.3%.  
     - \( \varepsilon 44 \): 2.9%.  

---

### Clinical Assessments

#### Protocol
- **Version**: OASIS3-v2.1 standardized assessments.  
- **Time Window**: ±14 days from baseline.  
- **Standards**: Aligned with NIH Common Data Elements (v2.0).

#### Quality Metrics
- **Completeness**: 89.7%.  
- **Accuracy**: 99.8%.  
- **Consistency**: 92%.  

---

### Neuroimaging Module

#### Modalities
1. **Structural MRI**: High-resolution T1-weighted imaging.  
2. **Functional MRI**: Resting-state connectivity mapping.  
3. **Diffusion Imaging**: White matter tractography.  

#### Processing
- Automated segmentation using **FreeSurfer** (v6.0).  
- Quality checks include intracranial volume normalization and artifact detection.

---

### Cognitive Assessments

#### Domains Evaluated
1. **Memory**: Logical Memory, Word Recall tests.  
2. **Executive Function**: Trail Making, Digit Span.  
3. **Processing Speed**: Visual reaction time tasks.  
4. **Metacognition**: Confidence ratings and metacognitive accuracy measures.

#### Assessment Metrics
- **Composite Scores**: Generated for each cognitive domain.  
- **Test-Retest Reliability**: >0.85 across domains.  

---

### Data Collection Workflow
1. **Participant Enrollment**: Assigned unique IDs upon consent.  
2. **Baseline Collection**: Combined sessions for cognitive testing and imaging.  
3. **Follow-Up Sessions**: Regular intervals for monitoring progression.  

---

## Dataset Components

### Overview
The MRN dataset comprises five primary sheets, each addressing a key data domain: demographics, neuroimaging, cognitive assessments, MRN metrics, and clinical outcomes.

---

### 1. Participant Demographics
#### Core Variables
- **Age**: Reported to two decimal places.  
- **Sex**: Coded as Male, Female, or Unknown.  
- **Education**: Completed academic years categorized into low, medium, and high.  
- **APOE Status**: Genotype categorization (\( \varepsilon 33 \), \( \varepsilon 34 \), etc.).  
- **Family History of AD**: Binary variable (Yes/No).  

#### Summary Statistics
```python
# Example Summary
Total_Participants = 2,681
Age_Stats = {'mean': 69.1, 'std': 9.1, 'range': [42.5, 95.6]}
Sex_Distribution = {'Female': 28.2%, 'Male': 23.2%, 'Unknown': 48.6%}
Education_Stats = {'mean': 15.8, 'std': 2.7, 'median': 16.0}
```

---

### 2. Neuroimaging Data
#### Key Features
- **Volume Metrics**:
  - Hippocampus Volume (normalized by intracranial volume).  
  - Entorhinal Cortex Volume.  
- **Connectivity Metrics**:
  - Functional connectivity between predefined regions.  
- **Processing Standards**:
  - FreeSurfer segmentation (v6.0).  
  - Normalization to MNI152 space.  

#### Example Data
```python
# Example Metrics
Hippocampus_Volume = {'mean': 0.179, 'std': 0.029, 'range': [0.065, 0.290]}
Memory_Network_Score = {'mean': 5879.03, 'std': 691.32}
```

---

### 3. Cognitive Assessments
#### Core Cognitive Domains
- **Memory**: Logical Memory, Free Recall.  
- **Executive Function**: Working memory and reasoning tasks.  
- **Processing Speed**: Trail Making Tests (A and B).  
- **Metacognition**: Accuracy and confidence ratings.  

#### Example Data
```python
# Example Metrics
MMSE_Score = {'mean': 28.22, 'std': 2.57, 'range': [16, 30]}
Memory_Recall_Score = {'mean': 21.36, 'std': 10.02}
```

---

### 4. MRN Metrics
#### Resilience Scores
- **Hippocampus Resilience**: Derived from memory performance and hippocampal volume.  
- **PFC Resilience**: Derived from executive function scores and PFC volume.  
- **Network Connectivity Score**: Weighted average of key network metrics.  
- **Compensation Index**: Accounts for non-structural factors influencing cognitive performance.  

#### Example Data
```python
# Example Metrics
Hippocampus_Resilience = {'mean': 88.94, 'std': 25.18, 'range': [0.00, 100.00]}
Overall_MRN_Score = {'mean': 47.76, 'std': 9.06}
```

---

### 5. Clinical Outcomes
#### Core Variables
- **Clinical Status**: Normal, MCI, or AD.  
- **Activities of Daily Living (ADL)**: Scored across tasks like managing finances and preparing meals.  
- **Behavioral Symptoms**: Binary coding of symptoms like anxiety and agitation.  
- **Treatment Response**: Categorized as Good, Moderate, or Poor.  

#### Example Data
```python
# Example Metrics
ADL_Scores = {'mean': 5.08, 'std': 8.62, 'range': [0, 90]}
Behavioral_Symptoms = {'Anxiety': 12.5%, 'Agitation': 8.9%}
```

---

### File Format and Storage
- **Format**: Excel workbook with separate sheets for each data domain.  
- **Metadata**: Included as a separate file for documentation purposes.  
- **Storage**: Secure cloud storage with restricted access.  

---

## Statistical Methods

### Overview
The MRN study employs a combination of descriptive, inferential, and multivariate statistical techniques to analyze relationships between cognitive performance, brain structure, and metacognitive resilience.

---

### 1. Descriptive Statistics
#### Purpose
Summarize central tendencies, variability, and distribution of key variables.

#### Methods
- **Central Tendency**: Mean, median, and mode.
- **Dispersion**: Standard deviation (SD), interquartile range (IQR).
- **Distribution Testing**: Shapiro-Wilk (\(W\)) and Kolmogorov-Smirnov tests.

#### Example Code
```python
import pandas as pd

# Compute descriptive statistics
summary = df.describe()
shapiro_w = stats.shapiro(df['Hippocampus_Volume'])
```

---

### 2. Group Comparisons
#### Purpose
Compare differences across groups based on clinical status or other categorical variables.

#### Methods
- **Continuous Variables**: ANOVA or t-tests for between-group differences.
- **Categorical Variables**: Chi-square tests or Fisher's exact test.
- **Post-hoc Analyses**: Tukey's HSD for multiple comparison corrections.

#### Example Code
```python
from scipy.stats import f_oneway, chi2_contingency

# ANOVA for hippocampal volume
anova_result = f_oneway(df[df['Group'] == 'AD']['Hippocampus_Volume'],
                        df[df['Group'] == 'MCI']['Hippocampus_Volume'],
                        df[df['Group'] == 'Normal']['Hippocampus_Volume'])

# Chi-square test for categorical data
chi2_result = chi2_contingency(pd.crosstab(df['Group'], df['APOE_Status']))
```

---

### 3. Correlation and Regression
#### Purpose
Examine relationships between structural brain metrics, cognitive performance, and MRN scores.

#### Methods
- **Pearson Correlation**: For linear relationships.
- **Partial Correlation**: Controlling for age, sex, and education.
- **Multiple Regression**: Predict cognitive outcomes using structural and connectivity metrics.

#### Example Code
```python
import statsmodels.api as sm

# Multiple regression
X = df[['Hippocampus_Volume', 'PFC_Volume', 'Connectivity_Score']]
y = df['Memory_Recall_Score']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

---

### 4. Longitudinal Analyses
#### Purpose
Evaluate changes over time in brain and cognitive metrics.

#### Methods
- **Mixed-Effects Models**: Account for repeated measures within participants.
- **Survival Analysis**: For progression to AD from MCI.

#### Example Code
```python
from statsmodels.regression.mixed_linear_model import MixedLM

# Mixed-effects model
model = MixedLM.from_formula("Cognitive_Score ~ Age + Timepoint", 
                              groups="Participant_ID", 
                              data=df)
result = model.fit()
print(result.summary())
```

---

### 5. Network Analysis
#### Purpose
Investigate connectivity patterns within and between brain networks.

#### Methods
- **Graph Theory**: Node centrality, clustering coefficients.
- **Functional Connectivity**: Correlation matrices of region pairs.
- **Network Resilience**: Weighted composite scores of network components.

#### Example Code
```python
import networkx as nx

# Create and analyze connectivity graph
G = nx.Graph()
G.add_weighted_edges_from(connectivity_data)
centrality = nx.betweenness_centrality(G)
```

---

### 6. Missing Data Treatment
#### Purpose
Handle incomplete data to ensure robust analyses.

#### Methods
- **Descriptive Analysis**: Assess missing data patterns.
- **Imputation**: Multiple imputation for key variables where missingness <20%.
- **Sensitivity Analysis**: Compare results with and without imputed data.

#### Example Code
```python
from sklearn.impute import SimpleImputer

# Imputation for missing data
imputer = SimpleImputer(strategy='mean')
df['Hippocampus_Volume'] = imputer.fit_transform(df[['Hippocampus_Volume']])
```

---

### 7. Multivariate Techniques
#### Purpose
Explore relationships among multiple variables simultaneously.

#### Methods
- **Principal Component Analysis (PCA)**: Reduce dimensionality.
- **Structural Equation Modeling (SEM)**: Test hypothesized relationships.

#### Example Code
```python
from sklearn.decomposition import PCA

# PCA for dimensionality reduction
pca = PCA(n_components=2)
components = pca.fit_transform(df[['Hippocampus_Volume', 'PFC_Volume', 'Connectivity_Score']])
```

---

## Quality Control

### Overview
Ensuring data accuracy, completeness, and consistency is critical for reliable results. The MRN project employs a multi-step quality control process to validate data integrity and identify potential errors.

---

### 1. Data Validation
#### Objectives
- Verify that all collected data adhere to predefined standards.
- Detect and correct outliers and inconsistencies.

#### Procedures
1. **Range Checks**:
   - Ensure values fall within anatomically and clinically plausible ranges.
   - Example: Hippocampal volume must be within 0.065–0.290.
2. **Missing Data Analysis**:
   - Identify missingness patterns.
   - Exclude or impute based on thresholds (<20% missing).
3. **Logical Consistency**:
   - Cross-validate related variables for consistency (e.g., APOE status vs. genetic data).

#### Example Code
```python
# Identify missing values
missing_summary = df.isnull().sum() / len(df) * 100

# Flag outliers
outliers = df[(df['Hippocampus_Volume'] < 0.065) | (df['Hippocampus_Volume'] > 0.290)]
```

---

### 2. Cleaning Procedures
#### Objectives
Standardize datasets and resolve discrepancies.

#### Methods
- **Standardization**: Ensure uniform formats (e.g., dates in YYYY-MM-DD format).
- **Outlier Treatment**:
  - Winsorization: Cap extreme values to 5th and 95th percentiles.
  - Exclusion: Remove values exceeding ±3 SD from the mean.
- **Data Transformation**:
  - Normalize volume metrics by intracranial volume.
  - Scale cognitive scores to z-scores for cross-domain comparison.

#### Example Code
```python
# Winsorize hippocampal volume
df['Hippocampus_Volume'] = df['Hippocampus_Volume'].clip(lower=df['Hippocampus_Volume'].quantile(0.05),
                                                          upper=df['Hippocampus_Volume'].quantile(0.95))

# Normalize volumes
df['Hippocampus_Volume_Normalized'] = (df['Hippocampus_Volume'] / df['Intracranial_Volume']) * 100
```

---

### 3. Quality Metrics
#### Metrics for Validation
1. **Completeness**:
   - Percentage of non-missing values.
   - Target: ≥95% for critical variables.
2. **Consistency**:
   - Intra-variable agreement (e.g., test-retest reliability).
   - Inter-variable logical alignment.
3. **Accuracy**:
   - Concordance between manually entered data and automated measures.

#### Example Output
```python
Completeness_Summary = {
    'Hippocampal_Volume': 98.5%,
    'Cognitive_Scores': 95.7%
}

Consistency_Checks = {
    'APOE_Status_vs_Genotype': 100%,
    'Hippocampal_Volume_vs_Age': 96%
}
```

---

### 4. Error Handling
#### Procedures
1. **Flagging**:
   - Automatically flag records for review if they fail validation checks.
2. **Resolution**:
   - Cross-reference flagged entries with source documents.
   - Implement corrections or exclude problematic entries.
3. **Documentation**:
   - Log all corrections and exclusions for reproducibility.

#### Example Code
```python
# Flag inconsistent data
df['Flagged'] = df['Hippocampus_Volume_Normalized'].apply(lambda x: True if x < 0 or x > 100 else False)

# Log corrections
correction_log = []
if len(outliers) > 0:
    for index, row in outliers.iterrows():
        correction_log.append(f"Corrected {row['Participant_ID']}: {row['Hippocampus_Volume']}")
```

---

### 5. Quality Assurance Audits
#### Frequency
- Initial: Conducted post-baseline data collection.
- Periodic: Every 6 months during the study.
- Final: Comprehensive audit before analysis.

#### Audit Checklist
1. Verify participant IDs match across all datasets.
2. Validate demographic distributions (e.g., age, sex ratios).
3. Recheck outlier handling and imputation methods.

---

### 6. Data Quality Reporting
#### Summary Reports
- Include statistics on completeness, flagged records, and error rates.
- Highlight key resolutions and remaining challenges.

#### Example Report
| Metric                  | Value       |
|-------------------------|-------------|
| Completeness            | 97.8%       |
| Flagged Records         | 1.2% (32/2681) |
| Outlier Rate (Volumes)  | 0.6%        |
| Logical Consistency     | 99.2%       |

---

## Technical Details

### Overview
This section outlines the software, hardware, and processing pipeline requirements for managing and analyzing data in the MRN project.

---

### 1. Software Requirements
#### Programming Languages and Libraries
- **Python**: \( \geq 3.8 \)
  - Core libraries: `pandas`, `numpy`, `scipy`, `statsmodels`
  - Neuroimaging libraries: `nibabel`, `nipype`
  - Machine learning: `scikit-learn`, `tensorflow` (optional for advanced analyses)
- **R**: \( \geq 4.0 \)
  - Libraries: `tidyverse`, `mice`, `lme4`, `semTools`

#### Data Visualization
- **Matplotlib** and **Seaborn**: Python-based plots.
- **ggplot2**: R-based plots.

#### Statistical Software
- **SPSS**: For advanced statistical analysis.
- **JASP**: Open-source alternative for Bayesian analyses.

#### Version Control
- **Git**: Version control for scripts and documentation.
- **Repository**: Hosted on a private GitHub repository.

---

### 2. Hardware Requirements
#### Computational Needs
- **Processor**: Minimum Intel i7 or equivalent.
- **RAM**: At least 16GB (32GB recommended for large neuroimaging datasets).
- **Storage**: Minimum 1TB, with additional backup storage of 2TB.
- **Graphics Processing Unit (GPU)**: Recommended for machine learning or intensive neuroimaging processing.

#### Data Servers
- **Cloud Storage**: Secure server hosted on AWS with restricted access.
- **Local Server**: RAID-configured storage for redundancy.

---

### 3. Data Processing Pipeline
#### Steps
1. **Data Ingestion**:
   - Load raw datasets from OASIS3 and other sources.
   - Validate file formats and structures.
2. **Preprocessing**:
   - Standardize demographic and clinical data.
   - Normalize neuroimaging volumes to intracranial volume (ICV).
3. **Feature Extraction**:
   - Compute network metrics and resilience scores.
   - Extract cognitive domain composites.
4. **Statistical Analysis**:
   - Perform group comparisons, correlation, and regression.
5. **Output Generation**:
   - Save cleaned datasets and analysis results.

#### Example Code for Pipeline
```python
def data_pipeline():
    # Step 1: Load raw data
    demographics = pd.read_csv("demographics.csv")
    neuroimaging = pd.read_csv("neuroimaging.csv")
    
    # Step 2: Preprocess
    demographics['Age'] = demographics['Age'].round(2)
    neuroimaging['Hippocampus_Volume_Normalized'] = (neuroimaging['Hippocampus_Volume'] / 
                                                     neuroimaging['ICV']) * 100
    
    # Step 3: Feature extraction
    resilience_scores = compute_resilience_scores(neuroimaging, demographics)
    
    # Step 4: Save outputs
    resilience_scores.to_csv("resilience_scores.csv", index=False)
```

---

### 4. Data Storage and Security
#### Storage Format
- **File Types**: Excel files with separate sheets for demographics, neuroimaging, cognitive assessments, and MRN metrics.
- **Compression**: Use `.zip` or `.gzip` for large files.

#### Security Protocols
- **Access Control**: Limited to approved researchers.
- **Encryption**: End-to-end encryption for all data transfers.
- **Backup**:
  - Daily incremental backups.
  - Weekly full backups stored offsite.

---

### 5. Quality Assurance Tools
#### Automated Scripts
- Validate file structures and formats.
- Check for missing data and outliers.
- Generate summary statistics.

#### Manual Checks
- Cross-verify automated outputs with raw datasets.
- Perform spot-checks on critical variables.

---

### 6. Version Control and Documentation
#### File Naming Convention
- **Format**: `<Dataset>_<Version>_<Date>.csv` (e.g., `Demographics_v1_20241203.csv`).
- **Versioning**: Increment versions with major updates (e.g., v1 → v2).

#### Change Log Example
| Date       | Version | Changes Made                                   |
|------------|---------|-----------------------------------------------|
| 2024-12-01 | v1.0    | Initial release of the demographics dataset.  |
| 2024-12-03 | v1.1    | Added APOE genotype to the demographics file. |

---

## Appendices

### Appendix A: List of Abbreviations
| Abbreviation | Full Form                          |
|--------------|------------------------------------|
| MRN          | Metacognitive Resilience Network  |
| AD           | Alzheimer’s Disease               |
| MCI          | Mild Cognitive Impairment         |
| APOE         | Apolipoprotein E                  |
| ICV          | Intracranial Volume               |
| MMSE         | Mini-Mental State Examination     |
| PFC          | Prefrontal Cortex                 |
| MRI          | Magnetic Resonance Imaging        |

---

### Appendix B: Data Dictionaries
#### Demographics Sheet
| Variable             | Description                                  | Data Type | Units     | Example Values   |
|----------------------|----------------------------------------------|-----------|-----------|------------------|
| Participant_ID       | Unique identifier                           | String    | -         | MRN001           |
| Age                  | Age at baseline                             | Float     | Years     | 68.5             |
| Sex                  | Biological sex                              | String    | -         | Male, Female     |
| Education_Years      | Total academic years completed              | Integer   | Years     | 16               |
| APOE_Status          | APOE genotype                               | String    | -         | ε33, ε34         |

#### Neuroimaging Sheet
| Variable               | Description                                | Data Type | Units     | Example Values   |
|------------------------|--------------------------------------------|-----------|-----------|------------------|
| Hippocampus_Volume     | Hippocampal volume                         | Float     | cm³       | 0.179            |
| Hippocampus_Normalized | Volume normalized to intracranial volume   | Float     | %         | 15.4             |
| Connectivity_Score     | Functional connectivity within networks    | Float     | -         | 0.85             |

---

### Appendix C: Software Installation Guide
#### Python Environment Setup
1. Install Python \( \geq 3.8 \):
   ```bash
   sudo apt-get install python3.8
   ```
2. Create a virtual environment:
   ```bash
   python3.8 -m venv mrn_env
   source mrn_env/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### R Environment Setup
1. Install R \( \geq 4.0 \):
   ```bash
   sudo apt-get install r-base
   ```
2. Install required libraries:
   ```R
   install.packages(c("tidyverse", "lme4", "mice"))
   ```

---

### Appendix D: Data Access Protocols
1. **Requesting Access**:
   - Submit a request to the project administrator.
   - Provide a description of intended use.
2. **Approval Process**:
   - Review by the data governance board.
   - Signing of a data-sharing agreement.
3. **Access Delivery**:
   - Secure download link provided after approval.
   - Access expires after a specified duration.

---

### Appendix E: Contact Information
- **Project Lead**: Robyn An
  - Email: rnhyunan@gmail.com; rna2118@barnard.edu
  - Department: Psychology, Barnard College of Columbia University  

---
