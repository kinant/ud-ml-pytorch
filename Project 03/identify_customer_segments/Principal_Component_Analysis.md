## Principal Component Analysis - Detailed Feature Interpretations

### PC1:

#### High Positive Scores

| Feature               | Weight  | Interpretation                                                                                       |
|-----------------------|---------|------------------------------------------------------------------------------------------------------|
| **FINANZ_MINIMALIST** | +0.2171 | Higher scores indicate lower minimalist tendency, meaning greater financial interest and engagement. |
| **MOBI_REGIO**        | +0.1972 | Lower movement patterns - people in that are less mobile and probably travel less frequently         |
| **PLZ8_ANTG1**        | +0.1952 | Higher proportion of 1-2 family houses in the broader PLZ8 region                                    |
| **KBA05_ANTG1**       | +0.1892 | Higher proportion of 1-2 family houses in the microcell - single-family home neighborhoods           |
| **KBA05_GBZ**         | +0.1884 | More buildings in the microcell                                                                      |

#### High Negative Scores

| Feature                  | Weight  | Interpretation                                                        |
|--------------------------|---------|-----------------------------------------------------------------------|
| **PLZ8_ANTG3**           | -0.1942 | Higher proportion of 6-10 family houses in PLZ8 region                |
| **CAMEO_WEALTH**         | -0.1874 | Lower wealth levels                                                   |
| **HH_EINKOMMEN_SCORE**   | -0.1869 | Lower household income                                                |
| **PLZ8_ANTG4**           | -0.1869 | Higher proportion of 10+ family apartment buildings in PLZ8 region    |
| **ORTSGR_KLS9**          | -0.1671 | Larger community size                                                 |

---

### PC2:

#### High Positive Scores

| Feature                    | Weight  | Interpretation                                                  |
|----------------------------|---------|------------------------------------------------------------------|
| **ALTERSKATEGORIE_GROB**   | +0.2202 | Older age groups                                                 |
| **FINANZ_VORSORGER**       | +0.2072 | "Be prepared" financial type - more prepared financially         |
| **ZABEOTYP_3**             | +0.2026 | "Fair supplied" energy consumers - fair energy consumption patterns |
| **SEMIO_ERL**              | +0.1820 | Event-oriented personality - higher affinity                     |
| **SEMIO_LUST**             | +0.1568 | Sensual-minded personality - lower affinity                      |

#### High Negative Scores

| Feature                     | Weight  | Interpretation                                                |
|-----------------------------|---------|---------------------------------------------------------------|
| **PRAEGENDE_DECADE**        | -0.2171 | Higher decade component - younger people                      |
| **FINANZ_UNAUFFAELLIGER**   | -0.2083 | Lower inconspicuous financial type - less secretivy about their money |
| **SEMIO_REL**               | -0.2075 | Religious personality - lower affinity                        |
| **FINANZ_SPARER**           | -0.2046 | Money-saver financial type - lower affinity                   |
| **SEMIO_TRADV**             | -0.2014 | Traditional-minded personality - lower affinity               |

---

### PC3: Gender & Personality Assertiveness

#### High Positive Scores

| Feature         | Weight  | Interpretation                       |
|-----------------|---------|--------------------------------------|
| **ANREDE_KZ**   | +0.3596 | More likely female gender            |
| **SEMIO_KAEM**  | +0.3294 | Lower combative attitude personality |
| **SEMIO_DOM**   | +0.2998 | Lower dominant-minded personality    |
| **SEMIO_KRIT**  | +0.2723 | Lower critical-minded personality    |
| **SEMIO_ERL**   | +0.1993 | Lower Event-oriented personality     |

#### High Negative Scores

| Feature          | Weight  | Interpretation                               |
|------------------|---------|----------------------------------------------|
| **SEMIO_VERT**   | -0.3347 | Lower dreamful personality                   |
| **SEMIO_FAM**    | -0.2623 | Lower family-minded personality              |
| **SEMIO_SOZ**    | -0.2619 | Lower socially minded personality            |
| **SEMIO_KULT**   | -0.2507 | Lower culturally minded personality          |
| **FINANZTYP_5**  | -0.1434 | Investor financial type (type 5 = ANLEGER)   |

---