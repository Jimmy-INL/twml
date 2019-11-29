---
marp: true
---

# DATA20019 Trustworthy Machine Learning

## Project 3, Firefighters

Eeva-Maria Laiho, 2.11.2019

---

# The Case: Ricci v. DeStefano

* In 2003 firefighters (n=118) took an exam to qualify for 15 promotions
    * The test was designed to be culturally unbiased
* No self-identified black candidates qualified
    * City officials feared lawsuit on the basis of disproportionate exclusion
    * The test was considered flawed and the results were invalidated
       => Promotions based on the exam results were cancelled
* 20 high-scoring firefighters filed a lawsuit on the basis a racial discrimination
* In 2009 U.S. Supreme Court ruled in favor of the plaintives (5-4 decision)
    * The court considered the test valid 
      => the highest-scoring firefighters were promoted

---

# The Data

* A data frame with 118 observations on the following 5 variables.
    * Race: Race of firefighter (B=black, H=Hispanic, or W=white)
    * Position: Promotion desired (Captain or Lieutenant)
    * Oral: Oral exam score
    * Written: Written exam score
    * Combine: Combined score (written exam gets 60% weight)

| Race | Position | Oral | Written | Combine |
| --- | --- | --- | --- | --- |
| H | Captain | 79.05 | 74 | 76.02 |
| W | Captain | 73.81 | 77 | 75.724 |
| W | Captain | 76.67 | 74 | 75.068 |
| B | Captain | 82.38 | 70 | 74.952 |
|...|

* https://rdrr.io/cran/Stat2Data/man/Ricci.html

---

# The Task

* Story: who gets promoted
* Model: linear regression


