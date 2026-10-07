# Healthcare Recommendation System using R

A machine-learning based web application built with **R + Shiny**. The user enters a
patient's age, gender and symptoms; the app predicts the most likely disease with a
Random Forest classifier and then shows a healthcare recommendation (specialist,
general information, dataset-based medicine information, home care and precautions).

> **Educational project only.** The prediction and the medicine information come from a
> stored dataset and a statistical model. They are **not** a medical diagnosis, a
> prescription or professional advice. Always consult a qualified doctor.

---

## 1. Project file structure

```
Healthcare-Recommendation/
├── app.R                    # ENTIRE application (UI + server + model + analytics)
├── healthcare_data.csv      # Demo dataset: 1,558 patient records x 32 columns
├── recommendations.csv      # 18 diseases x 9 columns (9th = Common_Symptoms, Causes)
├── generate_demo_data.R     # Optional: regenerates the two CSV files above
├── tests/
│   └── test_app.R           # Automated test of the whole workflow (93 checks)
└── README.md                # This file
```

Only **`app.R`** contains application code. The two CSV files are plain data.
`generate_demo_data.R` and `tests/test_app.R` are optional extras — the app runs
fine without them.

---
## Screenshots
![dashboard](<Screenshot 2026-10-07 190944.png>)
![disease prediction](<Screenshot 2026-10-07 191103.png>)
![doctor recommendation](<Screenshot 2026-10-07 191134.png>)
## 2. Installation (do this once)

Open R or RStudio and run **this one command**:

```r
install.packages(c("shiny", "dplyr", "ggplot2", "randomForest", "caret"))
```

| Package        | Why it is needed                                    | Required? |
|----------------|-----------------------------------------------------|-----------|
| `shiny`        | The web application / GUI                           | **Yes**   |
| `dplyr`        | Data processing (filter, group, count, mutate)       | **Yes**   |
| `ggplot2`      | The three analytics charts                          | **Yes**   |
| `randomForest` | The disease prediction model (300 decision trees)   | Recommended — without it the app falls back to a single decision tree (`rpart`, ships with R) |
| `caret`        | Stratified train/test split + confusion-matrix statistics | Optional — the app works without it |

The app checks for missing packages at startup and prints exact installation
instructions in the R console instead of crashing. Internet is needed only for
`install.packages()`; the app itself runs completely offline.

---

## 3. How to run

```r
setwd("path/to/Healthcare-Recommendation")   # the folder containing app.R
shiny::runApp()
```

In RStudio you can also just open `app.R` and click the **Run App** button.
The app opens in your browser (usually `http://127.0.0.1:3838`). Press `Esc` in the
R console, or click **Stop**, to close it.

### Running the app without `randomForest`
Nothing to change — the app detects it and automatically uses `rpart` instead, and
the Dashboard says which engine is in use.

### Regenerating the demo data (optional, not needed to run)
```r
setwd("path/to/Healthcare-Recommendation")
source("generate_demo_data.R")
```

### Running the automated tests (optional — useful as proof for your report)
```r
setwd("path/to/Healthcare-Recommendation")
source("tests/test_app.R")
```
This drives the Shiny server directly (no browser needed) and prints a PASS/FAIL
line for each of the **93 checks** covering data loading, preprocessing, prediction
quality, the recommendation lookup, and every error-handling path. Expected result:
`RESULT: 93 passed, 0 failed`.

---

## 4. Complete workflow of the project

```
        ┌──────────────────┐
        │  Patient Profile  │   age, gender, height/weight, habits, duration
        │  + Symptom Input  │   symptoms (with search)
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │  Preprocessing    │   28 symptoms coded 0/1 + age + gender
        │                   │   (29 features, exactly as in training)
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │   ML Model        │   Random Forest (300 trees), trained at startup
        │                   │   on a stratified 75 % split of the CSV
        └────────┬─────────┘
                 │  predicted disease + probability
        ┌────────▼─────────┐
        │ Recommenda-tion   │   rules: disease  ->  row in recommendations.csv
        │     Layer         │
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │     Output        │   disease, confidence, other possibilities,
        │                   │   specialist, medicine info, home care, precautions
        └──────────────────┘
```

### Step by step

1. **Data loading** — `healthcare_data.csv` and `recommendations.csv` are read with
   `read.csv()`. A small file-search helper finds them even if you launch the app from
   a parent folder. If a file is missing, a clear error message is printed.

2. **Data cleaning** (`clean_healthcare_data()`) — rows without a disease label are
   dropped; every symptom column is forced to integer `0`/`1` (missing → `0`);
   missing or impossible ages are replaced by the median age; missing/unknown genders
   are replaced by `"Male"`; and `Disease` is converted to a **factor** (essential,
   otherwise `randomForest` would run a regression instead of a classification).

3. **Feature engineering** (`prepare_features()`) — the user's input is turned into a
   one-row data frame with the **same 29 columns, in the same order** as the training
   data. This alignment is what makes prediction reliable.

4. **Model training** (`train_healthcare_model()`) — runs **once when the app starts**,
   not on every click, so the app stays fast during the demo.

5. **Prediction** (`predict_patient()`) — the trained model returns a probability for
   each of the 18 diseases; the disease with the highest probability becomes the
   prediction, and that probability is displayed as the **confidence**.

6. **Recommendation** (`get_recommendation()`) — a rule-based lookup of the predicted
   disease in `recommendations.csv`.

7. **Analytics** — three `ggplot2` charts plus a confusion matrix.

---

## 5. The machine-learning model explained

**Type of problem:** supervised **multi-class classification**
(18 possible disease classes).

| Item | Value |
|------|-------|
| Algorithm | Random Forest (`randomForest`, 300 trees) |
| Features (X) | 29 → 28 binary symptom columns + `Age` + `Gender` |
| Target (Y) | `Disease` (factor, 18 classes) |
| Split | Stratified 75 % train / 25 % test, `set.seed(123)`. With `caret`: 1,175 / 383. Without `caret` (base-R fallback): 1,162 / 396. |
| Training accuracy | ≈ 98.1 % |
| **Test accuracy** | **≈ 80–82 %** (81.6 % with the base-R split, 80.2 % with caret) |

### Why Random Forest?
* It is an **ensemble** of 300 decision trees, each grown on a bootstrap sample of the
  data with a random subset of features at each split. The trees vote and the majority
  vote wins — so single-tree overfitting is averaged out.
* It handles **many binary inputs** natively and needs **no feature scaling**.
* It gives **class probabilities** (used here as the confidence) and
  **variable importance** (shown in the Analytics tab).
* It is robust to irrelevant symptoms and to moderate class imbalance.

### Bias-variance / overfitting note
Training accuracy (≈ 98 %) is higher than test accuracy (≈ 81 %). This is expected and
is actually good evidence that the evaluation is honest: the test set contains records
the model has never seen. The gap is the classic **overfitting** signal, and it is
reduced by the averaging effect of many trees. (In a `rpart` single tree the gap is even
larger.)

### Probability and confidence
`predict(model, newdata, type = "prob")` returns one probability per disease. The
maximum is taken as the prediction, and it is reported as the confidence percentage.
A confidence below 40 % triggers a **low-confidence warning** in the UI, because it
means several diseases fit the given symptoms equally well.

---

## 6. The recommendation system explained

The recommendation layer is a **rule-based lookup table**, not a second ML model.
This keeps it transparent and easy to defend in a viva.

```
predicted disease  ──►  match on Disease column of recommendations.csv  ──►  show 6 fields
```

| Column in `recommendations.csv` | Shown as |
|---|---|
| `Specialist` | Recommended doctor (e.g. Pulmonologist) |
| `Description` | General information about the condition |
| `Medicines_Educational` | Medicine information — **labelled as educational / dataset-based, not a prescription** |
| `Home_Care` | Self-care advice |
| `Common_Symptoms` | Typical symptom list shown on the disease information card |
| `Causes` | Common causes of the disease |
| `Precautions` | When to see a doctor |
| `Urgency_Level` | Colour-coded badge: green = routine, amber = soon, red = urgent/emergency |

**Safety fallbacks**
* If a disease has no row in the file, a generic "General Physician" recommendation is
  shown instead of an error.
* `rec_field()` returns a safe default for any missing or blank column, so a renamed or
  absent column can never crash the app.
* The medical disclaimer is printed on the Dashboard, on the Results area and in the
  Help tab.

---

## 7. Features checklist

### v1 core requirements

| Requirement | Where it is implemented |
|---|---|
| **Dashboard** — title, description, #diseases, #records, accuracy | Dashboard tab: hero banner + 4 stat cards + workflow steps + model info + "What's included" |
| **Patient input** — age, gender, multiple symptoms | Predict tab sidebar |
| **Disease prediction** — Random Forest + confidence | `train_healthcare_model()`, `predict_patient()` |
| **Recommendation** — specialist, info, medicines | `get_recommendation()`, rendered in the result cards |
| **Analytics** — 3 ggplot2 charts | Disease distribution, most common symptoms, per-disease accuracy (+ confusion matrix + importance) |
| **UI** — clean healthcare theme, cards, responsive | Custom CSS in `APP_CSS`, Bootstrap grid, navbar navigation |

### v2 enhancements

| # | Feature | Where it is implemented |
|---|---|---|
| 1 | **Patient profile** — age, gender, height, weight, live BMI + band, smoking, alcohol, existing conditions, symptom duration | Predict tab sidebar; `bmi_value()`, `bmi_band()`, `read_patient_inputs()`; repeated in the result as "Patient profile recorded" |
| 2 | **Symptom search** — live filter over 28 symptoms, with selected-symptom chips | `visible_choices()`, `current_symptoms()`, `output$selected_chips` |
| 3 | **Prediction with confidence** | `predict_patient()`, colour-coded confidence bar |
| 4 | **Top 3 possible diseases** — each with probability and match score | `.hc-top3` cards in `output$result_ui` |
| 5 | **Specialist recommendation** | `recommendations.csv` → `rec_field(rec, "Specialist")` |
| 6 | **Educational medicine information** | Labelled disclaimer block; also in the HTML report |
| 7 | **Symptom-to-disease matching score** | `build_symptom_profile()` → `match_disease_to_symptoms()`; per-symptom table with Typical / Sometimes seen / Not typical |
| 8 | **Disease information card** — symptoms, causes, precautions, general info | New `Common_Symptoms` + `Causes` columns; also a standalone **disease information browser** on the History tab that works without predicting |
| 9 | **Analytics dashboard** — distribution, symptom frequency, accuracy | Analytics tab (`ggplot2`) |
| 10 | **Prediction history for the session** | `history_df` reactiveVal; History tab with 4 summary stat cards and a table |
| 11 | **Download as CSV / report** | `download_history` (CSV), `download_prediction` (CSV), `download_report` (HTML → print to PDF) |
| 12 | **Validation for missing symptoms and invalid inputs** | `read_patient_inputs()` collects **all** problems and lists them together |
| 13 | **Professional healthcare UI** | Cards, Font Awesome icons, responsive grid, `@media` rules for narrow screens |
| 14 | **Reset + New Patient** | `reset_btn` clears symptoms only (keeps the profile); `new_patient_btn` clears everything |
| — | **Load Sample Patient** (bonus, for demos) | `sample_btn` fills a complete Dengue-like case in one click |

### Why the v2 additions are safe additions, not a rewrite

* `healthcare_data.csv` was **not changed at all** — the byte-for-byte md5 hash is
  identical, so the model, the accuracy and every existing chart are unchanged.
* `recommendations.csv` only **gained** two columns (`Common_Symptoms`, `Causes`).
  `rec_field()` already returned a safe default for missing columns, so the app
  still works if you supply an older 7-column file.
* The training/evaluation path, the prediction path and the recommendation path
  are the same functions as before.

### Input validation & error handling (all tested)

* No symptoms selected → friendly warning card, no crash.
* Age empty / non-numeric / out of 1–120 → clear validation message.
* Height, weight and symptom duration validated to plausible ranges.
* **Several problems at once** are shown as a bulleted list instead of one at a time.
* Gender / smoking / alcohol / conditions have safe defaults if unset.
* `randomForest` not installed → automatic `rpart` fallback.
* `caret` not installed → base-R stratified split, and the Kappa card says so.
* Model training failed → Dashboard shows the reason instead of breaking.
* CSV file missing/empty → clear `[FILE NOT FOUND]` message.
* Prediction throws an error → caught by `tryCatch()` and shown as a message card.
* Missing values in the data → replaced during cleaning.
* Unseen/renamed symptom columns → handled by `prepare_features()`.
* Low model confidence (< 40 %) → warning banner suggesting more symptoms.
* Empty history → "no predictions yet" empty state; downloading an empty history
  produces a CSV explaining that, rather than an error.
* Symptoms ticked under a search filter are **not** silently dropped when the
  search text changes (this was a real bug found by testing — see section 12).

---

## 8. Using your own dataset

1. Replace `healthcare_data.csv`, keeping these column names:

   | Column | Type | Meaning |
   |---|---|---|
   | `PatientID` | text | optional ID |
   | `Age` | number | years |
   | `Gender` | text | `Male` / `Female` |
   | 28 symptom columns | 0 / 1 | 1 = symptom present |
   | `Disease` | text | the label the model predicts |

2. If your symptom column names are different, edit the `SYMPTOM_LABELS` vector at the
   top of `app.R` — that vector is the **single source of truth**; the checkbox list,
   the feature matrix and the charts all read from it.

3. If your disease names are different, update the `Disease` column of
   `recommendations.csv` to match. Any disease you leave out simply shows the safe
   generic recommendation.

   The two columns added in v2 (`Common_Symptoms`, `Causes`) are **optional** —
   if they are missing, the disease information card shows
   "Not available in the dataset." instead of failing, and the rest of the app is
   unaffected.

4. Restart the app. The model retrains automatically.

Good public sources: **Kaggle — "Disease Symptom Prediction"**, **UCI ML Repository —
Disease datasets**, or any hospital/OPD export you are permitted to use.

---

## 9. Viva questions and answers

**Q1. Why did you use Random Forest instead of logistic regression or a single decision tree?**
Because the target has 18 classes and the inputs are 28 binary symptoms plus age, which
produce non-linear, interacting patterns. A single decision tree overfits badly and gives
no probabilities. Random Forest builds 300 trees on bootstrap samples with random feature
subsets and averages their votes, which reduces variance, resists overfitting, needs no
feature scaling, and also gives per-class probabilities (used here as the confidence) and
variable importance. Logistic regression would also need 17 separate one-vs-rest models
for 18 classes and would struggle with symptom interactions.

**Q2. What is your train-test split and why is the test accuracy lower than the training accuracy?**
A stratified 75/25 split with a fixed seed (`set.seed(123)`) gives roughly 1,160–1,180
training records and 380–400 test records, depending on whether `caret` is installed. Stratification keeps every disease represented in both parts even though the
classes are imbalanced. Training accuracy (≈ 98 %) is higher than test accuracy (≈ 81 %)
because the trees have partly memorised the training records — the classic
overfitting/bias-variance gap. The test accuracy is the honest figure because those test
records were never seen during training. The averaging over 300 trees keeps the gap far
smaller than it would be for a single tree.

**Q3. How does your system produce a confidence value, and what do you do when it is low?**
`predict(model, newdata, type = "prob")` returns a probability for each of the 18 diseases
based on the proportion of trees voting for that class. The maximum probability is the
prediction and is displayed as the confidence percentage with a colour-coded bar
(green ≥ 70 %, amber ≥ 40 %, red below). If it is below 40 % the app shows a
low-confidence warning and suggests adding more distinctive symptoms, because a low value
means several diseases fit the symptoms equally well — so the result must not be trusted
blindly.

**Q4. What exactly is the "recommendation system" and how is it connected to the model?**
It is a rule-based lookup, deliberately kept separate from the ML model so it is
transparent and auditable. The predicted disease label is used as a key to fetch the
matching row of `recommendations.csv`, which supplies the specialist, the condition
description, the general medicine information, home care, precautions and an urgency
level. It is **not** a second model and **not** generated by AI — meaning every
recommendation can be traced back to a specific row that a human wrote. If a disease has
no row, the app falls back to a generic "General Physician" recommendation rather than
failing.

**Q5. How do you handle missing values, invalid input and the case where the user selects no symptoms?**
Three layers. (1) **Data level:** `clean_healthcare_data()` drops records with no disease
label, converts every symptom column to integer 0/1 with `NA → 0`, replaces missing or
out-of-range ages with the median age, and replaces missing/unknown genders with the most
common value. (2) **Input level:** the Predict button checks that the age is numeric and
between 1 and 120 and that at least one valid symptom is ticked; otherwise a friendly
message card is shown and no prediction is attempted. (3) **Runtime level:** the split,
the training and the prediction are each wrapped in `tryCatch()`, and `rec_field()` returns
a safe default for any missing recommendation column — so the app degrades gracefully
instead of crashing during a demonstration.

---

### Three more questions about the v2 features

**Q6. How is the symptom-matching score different from the model probability, and why show both?**
They answer different questions. The **model probability** is the Random Forest's vote across 300 trees — it uses all 29 features at once and can pick up patterns like "fever + joint pain + rash + age 34 ⇒ Dengue". The **match score** is transparent and per-symptom: it averages P(symptom | disease) taken straight from the dataset, i.e. "what share of the patients who actually have Dengue also reported this symptom". Showing both is a useful sanity check — a high probability with a low match score means the model is relying on something other than the symptoms you ticked, which is a good reason for the user to be cautious. The probability is the prediction; the match score is the explanation.

**Q7. Where is the session history stored, and what happens to it?**
In a `reactiveVal` called `history_df` inside the server function — that is, in the RAM of the running R session only. Nothing is written to disk and there is no database, which is exactly what the project constraints require. The history therefore belongs to one running session: refreshing the browser keeps it, but stopping and restarting the app clears it, and each browser session gets its own copy. If you wanted it to persist you would append to a CSV with `write.table(..., append = TRUE)`, but for a demonstration project the in-memory version is safer and simpler.

**Q8. Why is the report downloaded as HTML rather than PDF?**
Because a real PDF needs a LaTeX or Pandoc installation sitting next to R, and that is the single most common reason a student's "download PDF" button fails on the examiner's laptop. Instead `build_report_html()` writes one self-contained HTML file using the `build_report_html()` function — it embeds its own CSS, needs no internet, and opens in any browser. From there **Print → Save as PDF** produces a proper PDF, including the tables and page-break rules that are already defined in the report's print stylesheet. The report builder is a plain function rather than inline code inside the download handler, so it can be unit-tested without a browser.

---

## 10. How this project was tested

Two independent test layers were used, and both are reproducible.

**1. R test suite — `tests/test_app.R` (93 checks)**

```r
setwd("path/to/Healthcare-Recommendation")
source("tests/test_app.R")          # expect: RESULT: 93 passed, 0 failed
```

It uses `shiny::testServer()` to drive the real server function without a browser,
covering data loading, preprocessing, prediction quality, the new v2 helpers
(symptom matching, BMI, report building), every validation path, the history, the
download handlers and the reset/new-patient behaviour.

**2. Real browser walkthrough (62 checks)**

The application was driven in a headless Chromium browser: filling the patient
profile, filtering the symptom list, ticking symptoms, predicting, reading the
Top 3 cards and the information card, hitting each validation error, loading the
sample patient, checking the history table, downloading all three files, and
rendering all three `ggplot2` charts. Result: **62 passed, 0 failed, 0 browser
console errors**.

### Two real bugs that testing caught

Both were in the new symptom-search feature, and both would have been visible to
an examiner:

1. **The prediction ignored the ticked symptoms.** `do_predict()` read a cached
   copy of the selection that was filled in by a separate observer. In the browser
   the observer always runs first so it appeared to work, but the logic was
   fragile. Fixed by reading the symptoms **synchronously** through
   `current_symptoms()`.
2. **Selections were compared using display labels instead of keys.** The symptom
   choices are a named vector — names are labels (`"Joint Pain"`) and values are
   keys (`"joint_pain"`). A `checkboxGroupInput` reports **keys**, so comparing
   against `names()` never matched. The effect: clearing the search box wiped the
   user's selection, and a cleared selection came back on the next prediction.
   Fixed in three places by matching against `unname(...)`. Regression tests were
   added so it cannot come back.

This is worth mentioning in your viva: it shows the difference between "the test
passed" and "the application is correct".

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `there is no package called 'shiny'` | Run `install.packages(c("shiny","dplyr","ggplot2","randomForest","caret"))` |
| `[FILE NOT FOUND] healthcare_data.csv` | `setwd()` into the folder that contains `app.R` and the CSV files, then `shiny::runApp()` |
| Port already in use | `shiny::runApp(port = 3839)` |
| App is slow on the first click | The Random Forest trains at startup; the console prints the accuracy. Subsequent predictions are instant. |
| Charts are empty | Check the console — the app prints the number of records, diseases and the accuracy at startup. |
| Encoding/garbled characters | Save `app.R` as UTF-8 (RStudio: File → Save with Encoding → UTF-8). |
| Download buttons do nothing | They only appear on the **History** tab. A CSV download of an empty history still works and explains that no predictions were made. |
| Report opens but prints oddly | Use the browser's Print → Save as PDF. The report already has print styles; set margins to "Default" and enable "Background graphics". |
| BMI shows "--" | Height or weight is outside the accepted range (30–250 cm, 5–300 kg). |
| Symptom list looks short | The search box is filtering it. Clear the search text to see all 28 symptoms. |
| Packages were reinstalled but the app still fails | Make sure you launched R from the folder containing `app.R`: `setwd()` then `shiny::runApp()`. |

---

## 12. Technology constraints respected

✅ R only  · ✅ Shiny GUI  · ✅ CSV storage  · ✅ `dplyr`  · ✅ `ggplot2`
✅ `randomForest` (+ `caret`)  · ✅ All code in `app.R`
✅ No database · ✅ No Python · ✅ No React/Node.js · ✅ No login · ✅ No external API
