# MRN Project Documentation
## Table of Contents
1. [Project Overview](#project-overview)
2. [Study Design](#study-design)
3. [Data Collection Methods](#data-collection-methods)
4. [Dataset Components](#dataset-components)
5. [Statistical Methods](#statistical-methods)
6. [Quality Control](#quality-control)
7. [Technical Details](#technical-details)

## Project Overview
### Purpose
Exploration of the Metacognitive Resilience Network (MRN) in neurodegenerative diseases, focusing on early-stage Alzheimer's Disease.

### Key Features
- Integration of multimodal neuroimaging data
- Comprehensive cognitive assessment analysis
- Clinical outcome tracking
- Automated data processing pipeline

## Study Design
### Population Characteristics
#### Inclusion Criteria
- Age: 40-100 years
- Cognitive Status: Normal to moderate AD
- Consent: Full capacity required

#### Exclusion Criteria
- Severe cognitive impairment
- Non-AD neurological conditions
- MRI contraindications

### Sampling Framework
- **Method**: Community and clinic-based recruitment
- **Location**: US Midwest region
- **Statistical Power**: 95% for effect size d=0.3 (α=0.05)

## Data Collection Methods
### Demographics Module
#### Core Variables
1. **Participant ID (MRNxxx)**
   - Format: 3-digit unique identifier
   - Error rate: <0.001%
   - Validation: Checksum verification

2. **Age**
   - Resolution: 0.01 years
   - Range: 42.5-95.6 years
   - Distribution: Normal (W=0.994)

3. **Education**
   - Measure: Completed academic years
   - Reliability: ICC=0.97
   - Categories:
     - Low: <12 years (15.3%)
     - Medium: 12-16 years (52.7%)
     - High: >16 years (32.0%)

4. **APOE Status**
   - Method: PCR + HhaI restriction
   - Quality: 5% duplicate testing
   - Distribution:
     ```
     ε33: 24.1% (n=647)
     ε34: 16.6% (n=445)
     ε23: 5.3%  (n=143)
     ε44: 2.9%  (n=78)
     ```

### Clinical Assessments
#### Protocol Details
- Version: OASIS3-v2.1
- Window: ±14 days from baseline
- Standard: NIH Common Data Elements v2.0

#### Quality Metrics
- Completeness: 89.7%
- Accuracy: 99.8%
- Consistency: 0.92

## Dataset Components
### 1. Demographics Summary
```python
# Population Overview
Total_Participants = 2,681
Age_Stats = {
    'mean': 69.1,
    'sd': 9.1,
    'range': [42.5, 95.6]
}
Sex_Distribution = {
    'Female': 756,  # 28.2%
    'Male': 622,    # 23.2%
    'Unknown': 1303 # 48.6%
}
```

### 2. Education Distribution
```python
Education_Stats = {
    'mean': 15.8,
    'sd': 2.7,
    'median': 16.0,
    'range': [6.0, 29.0]
}
```

## Statistical Methods
### Primary Analyses
1. **Descriptive Statistics**
   - Central tendency (mean, median)
   - Dispersion (SD, IQR)
   - Distribution tests

2. **Group Comparisons**
   - ANOVA (continuous variables)
   - Chi-square (categorical)
   - Multiple comparison corrections

3. **Missing Data Treatment**
   ```R
   # Missing Pattern Analysis
   library(mice)
   md.pattern(demographics_data)
   ```

## Quality Control
### Data Validation
1. **Entry Verification**
   - Double-entry system
   - Automated range checks
   - Logic validation

2. **Cleaning Procedures**
   - Standardization
   - Outlier detection (Tukey)
   - Source verification

### Quality Metrics
```python
def quality_assessment(data):
    return {
        'completeness': data.notna().mean(),
        'validity': check_range_validity(data),
        'consistency': check_internal_consistency(data)
    }
```

## Technical Details
### Version Information
- Version: 1.0.0
- Release: December 2024
- Status: Initial Release

### Dependencies
- Python 3.8+
- R 4.0+
- Required libraries listed in requirements.txt
