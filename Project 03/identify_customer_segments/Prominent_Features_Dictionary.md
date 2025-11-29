# Data Dictionary for Prominent Features in Customer Segments Data

## 1. Person-level features

### 1.1. FINANZ_MINIMALIST, FINANZ_SPARER,
###      FINANZ_UNAUFFAELLIGER
Financial typology, for each dimension:
- -1: unknown
-  1: very high
-  2: high
-  3: average
-  4: low
-  5: very low

Dimension translations:
- MINIMALIST: low financial interest
- SPARER: money-saver
- UNAUFFAELLIGER: inconspicuous

### 1.2. PRAEGENDE_JUGENDJAHRE
Dominating movement of person's youth (avantgarde vs. mainstream; east vs. west)
### PRAEGENDE_DECADE:
- 1940 to 1990

### 1.3. LP_LEBENSPHASE_FEIN
Life stage, fine scale
### LP_AGE_CLASS
-  1 Younger age 
-  2 Middle age
-  3 Higher age
-  4 Advanced age
-  5 Retirement age
-----
## 2. Household-level features

### 2.1. HH_EINKOMMEN_SCORE
Estimated household net income
- -1: unknown
-  0: unknown
-  1: highest income
-  2: very high income
-  3: high income
-  4: average income
-  5: lower income
-  6: very low income

-----

## 4. RR4 micro-cell features

### 4.1. CAMEO_INTL_2015
German CAMEO: Wealth / Life Stage Typology, mapped to international code

-----

## 5. RR3 micro-cell features

### 5.1. KBA05_ANTG1
Number of 1-2 family houses in the microcell
- -1: unknown
-  0: no 1-2 family homes
-  1: lower share of 1-2 family homes
-  2: average share of 1-2 family homes
-  3: high share of 1-2 family homes
-  4: very high share of 1-2 family homes

-----
## 7. RR1 region features

### 7.1. MOBI_REGIO
Movement patterns
- 1: very high movement
- 2: high movement
- 3: middle movement
- 4: low movement
- 5: very low movement
- 6: none

## 8. PLZ8 macro-cell features

### 8.1. PLZ8_ANTG3
Number of 6-10 family houses in the PLZ8 region
- -1: unknown
-  0: no 6-10 family homes
-  1: lower share of 6-10 family homes
-  2: average share of 6-10 family homes
-  3: high share of 6-10 family homes


