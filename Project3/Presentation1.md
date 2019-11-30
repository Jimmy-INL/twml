---
marp: true
theme: presentation
---

# DATA20019 Trustworthy Machine Learning

## Project 3, Firefighters

Eeva-Maria Laiho, 2.11.2019

---

# Case: Firefighter Promotions

* New Haven FD administered an exam for firefighters to apply for promotion 
* Score of 70% or higher was required to pass for promotion
* Exam results would be valid for promoting for the next 2 years 
* City charter required that when k promotions are made the promotees must be selected from the k+2 top scorers (of the exam)
* At the time there were: 8 open positions for Lieutenant, 7 for Captain
* Within 2 years a total of 16 Lieutenant and 8 Captain positions became available
* Total of 118 firefighters took the exam

---

# Exam Results

![Exam results width:700px](./exam_results.png)

<small>Miao, W. (2010). Did the results of promotion exams have a disparate impact on minorities? Using statistical evidence in Ricci v. DeStefano. Journal of Statistics Education, 18(3).</small>

---

# Lawsuit: Ricci v. DeStefano

* City of New Haven decided not to certify the exam and no one was promoted
    * Insufficient number of minorities would be getting a promotion 
* The high-scoring test takers filed a lawsuit against the city on the grounds of reverse discrimination
* Both District Court and Trial Court decided in favor of city of New Haven
    * On the basis of "four-fifths rule" and adverse impact ratio 
* Supreme Court ruled in favor of the firefighters
    * The exam results had to be certified



---

# Data

* 118 observations, 5 variables
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

---

# Discriminatory measures (original data)

* Adverse impact ratio (pass rate)
    * For Lieutenant:
        * 31.6% / 58.1% = 54% (black / white)
        * 20% / 58.1% = 34.4% (hispanic / white)
    * For Captain:
        * 37.5% / 64% = 58.6% (black / white)
        * 37.5% / 64% = 58.6% (hispanic / white)
* All are below 80% from the guideline


---

# Task

* Predict who is promoted now (8+7 positions) / within 2 years (16+8 positions)
    * Logistic Regression model
    * 70-30 train-test split

* Compute discriminatory measures for predicted data
    * Adverse impact ratio


