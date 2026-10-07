# ============================================================================
#  tests/test_app.R
#  ---------------------------------------------------------------------------
#  Automated test of the whole workflow, using shiny::testServer().
#  It does NOT need a browser - it drives the server function directly.
#
#  RUN WITH:
#      setwd("path/to/Healthcare-Recommendation")
#      source("tests/test_app.R")
#
#  This file is NOT needed to run the application. It exists so you can prove
#  (in your report / viva) that the application was actually tested.
# ============================================================================

if (!requireNamespace("shiny", quietly = TRUE)) {
  stop("Package 'shiny' is required to run the tests.")
}

# --- locate app.R -----------------------------------------------------------
app_path <- file.path("app.R")
if (!file.exists(app_path)) app_path <- file.path("..", "app.R")
if (!file.exists(app_path)) stop("app.R was not found. Run this from the project folder.")

cat("\n============================================================\n")
cat(" TESTING: Healthcare Recommendation System\n")
cat("============================================================\n")

# sourcing app.R also builds the UI, the data and the trained model
source(app_path, local = FALSE)

pass <- 0L
fail <- 0L
check <- function(label, condition) {
  ok <- isTRUE(condition)
  if (ok) pass <<- pass + 1L else fail <<- fail + 1L
  cat(sprintf("  [%s] %s\n", if (ok) "PASS" else "FAIL", label))
}

html_of <- function(x) paste(as.character(x), collapse = "")

# ============================================================================
# TEST GROUP 1 - data + model loaded correctly
# ============================================================================
cat("\n-- Group 1: data and model --\n")

check("healthcare_data.csv loaded with records", nrow(HEALTHCARE_DATA) > 0)
check("all 28 symptom columns are 0/1",
      all(vapply(HEALTHCARE_DATA[, SYMPTOM_COLS],
                 function(x) all(x %in% c(0L, 1L)), logical(1))))
check("Disease column is a factor",  is.factor(HEALTHCARE_DATA$Disease))
check("no NA left in Age",           !any(is.na(HEALTHCARE_DATA$Age)))
check("no NA left in Gender",        !any(is.na(HEALTHCARE_DATA$Gender)))
check("recommendations.csv loaded",  nrow(RECOMMENDATIONS) > 0)
check("every disease has a recommendation row",
      length(setdiff(DISEASE_LEVELS, as.character(RECOMMENDATIONS$Disease))) == 0)
check("model trained without error", isTRUE(MODEL_INFO$trained_ok))
check("test accuracy is between 0 and 1",
      is.finite(MODEL_INFO$test_acc) && MODEL_INFO$test_acc > 0 && MODEL_INFO$test_acc < 1)

cat(sprintf("       -> records=%d  diseases=%d  symptoms=%d  test accuracy=%.2f%%\n",
            nrow(HEALTHCARE_DATA), length(DISEASE_LEVELS), length(SYMPTOM_COLS),
            100 * MODEL_INFO$test_acc))

# ============================================================================
# TEST GROUP 2 - preprocessing helpers
# ============================================================================
cat("\n-- Group 2: preprocessing and helpers --\n")

mk_patient <- function(age, gender, syms) {
  p <- as.data.frame(matrix(0L, nrow = 1L, ncol = length(SYMPTOM_COLS),
                            dimnames = list(NULL, SYMPTOM_COLS)),
                     stringsAsFactors = FALSE)
  if (length(syms) > 0) p[1, syms] <- 1L
  p$Age <- age; p$Gender <- gender
  prepare_features(p)
}

f <- mk_patient(34, "Female", c("fever", "joint_pain", "body_pain", "headache"))
check("prepare_features returns the correct predictor columns",
      identical(names(f), MODEL_VARS))
check("prepare_features codes selected symptoms as 1",
      all(f[, c("fever", "joint_pain", "body_pain", "headache")] == 1))
check("prepare_features codes unselected symptoms as 0",
      all(f[, setdiff(SYMPTOM_COLS, c("fever", "joint_pain", "body_pain", "headache"))] == 0))

f_na <- mk_patient(NA, NA, character(0))
check("NA age is replaced by a default (no crash)", is.finite(f_na$Age))
check("NA gender is replaced by a valid level",      as.character(f_na$Gender) %in% c("Male", "Female"))

# ============================================================================
# TEST GROUP 3 - prediction quality
# ============================================================================
cat("\n-- Group 3: prediction --\n")

predict_case <- function(age, gender, syms) {
  predict_patient(MODEL_INFO, mk_patient(age, gender, syms))
}

r1 <- predict_case(28, "Male",   c("sneezing", "runny_nose", "sore_throat", "cough"))
r2 <- predict_case(34, "Female", c("fever", "joint_pain", "body_pain", "headache", "skin_rash"))
r3 <- predict_case(56, "Male",   c("frequent_urination", "excessive_thirst", "fatigue"))
r4 <- predict_case(45, "Male",   c("cough", "night_sweats", "weight_loss", "fever"))

check("cold symptoms predict Common Cold / Allergic Rhinitis",
      r1$disease %in% c("Common Cold", "Allergic Rhinitis"))
check("dengue-like symptoms predict Dengue",             r2$disease == "Dengue")
check("diabetes-like symptoms predict Diabetes Mellitus", r3$disease == "Diabetes Mellitus")
check("TB-like symptoms predict Tuberculosis",            r4$disease == "Tuberculosis")
check("probability is between 0 and 1",
      all(vapply(list(r1, r2, r3, r4), function(x)
        x$confidence >= 0 && x$confidence <= 1, logical(1))))
check("top-3 ranking has 3 rows and covers the top disease",
      nrow(r2$ranking) == 3 && r2$ranking$Disease[1] == r2$disease)
check("recommendation returns all 7 fields",
      all(c("Disease", "Specialist", "Description", "Medicines_Educational",
            "Home_Care", "Precautions", "Urgency_Level") %in%
            names(get_recommendation(r2$disease))))
check("unknown disease falls back to General Physician",
      rec_field(get_recommendation("ZZZ Not A Disease"), "Specialist") == "General Physician")

# ============================================================================
# TEST GROUP 4 - v2 helpers: symptom matching, BMI, report building
# ============================================================================
cat("\n-- Group 4: v2 helper functions --\n")

check("SYMPTOM_PROFILE has one row per disease and one column per symptom",
      nrow(SYMPTOM_PROFILE) == length(DISEASE_LEVELS) &&
      ncol(SYMPTOM_PROFILE) == length(SYMPTOM_COLS))
check("SYMPTOM_PROFILE holds probabilities in [0,1]",
      all(SYMPTOM_PROFILE >= 0 & SYMPTOM_PROFILE <= 1))
check("P(fever|Dengue) is high",
      SYMPTOM_PROFILE["Dengue", "fever"] > 0.8)
check("P(cough|Dengue) is low",
      SYMPTOM_PROFILE["Dengue", "cough"] < 0.2)

m_ok <- match_disease_to_symptoms("Dengue",
          c("fever", "joint_pain", "body_pain", "headache", "skin_rash"))
m_bad <- match_disease_to_symptoms("Anemia",
          c("fever", "joint_pain", "body_pain", "headache", "skin_rash"))
check("matching score is higher for the right disease than a wrong one",
      m_ok$score > m_bad$score)
check("matching detail has one row per selected symptom",
      nrow(m_ok$detail) == 5)
check("matching counts add up to the number of symptoms",
      (m_ok$matched + m_ok$partial + m_ok$unmatched) == 5)
check("empty symptom list gives NA score, not an error",
      is.na(match_disease_to_symptoms("Dengue", character(0))$score))
check("unknown disease gives NA score, not an error",
      is.na(match_disease_to_symptoms("Nope", c("fever"))$score))
check("match sentence never claims more than was given",
      grepl("^1 of your 1 ", match_sentence(m_ok, 1)))

check("BMI computed correctly (165cm/62kg == 22.8)",
      isTRUE(all.equal(bmi_value(165, 62), 22.8, tolerance = 0.05)))
check("BMI is NA for impossible height", is.na(bmi_value(0, 60)))
check("BMI bands classify correctly",
      bmi_band(16)$label == "Underweight" && bmi_band(22)$label == "Normal" &&
      bmi_band(27)$label == "Overweight" && bmi_band(35)$label == "Obese")

check("new Common_Symptoms column has content for every disease",
      all(nzchar(vapply(seq_len(nrow(RECOMMENDATIONS)), function(i)
        rec_field(as.list(RECOMMENDATIONS[i, , drop = FALSE]), "Common_Symptoms", ""),
        character(1)))))
check("new Causes column has content for every disease",
      all(nzchar(vapply(seq_len(nrow(RECOMMENDATIONS)), function(i)
        rec_field(as.list(RECOMMENDATIONS[i, , drop = FALSE]), "Causes", ""),
        character(1)))))
check("esc() escapes HTML metacharacters",
      identical(esc("a<b>&c"), "a&lt;b&gt;&amp;c"))
check("report_bullets() builds a list from a ';' field",
      grepl("<ul>.*<li>One</li>.*</ul>", report_bullets("One; Two")))

# ============================================================================
# TEST GROUP 5 - the Shiny server itself
# ============================================================================
cat("\n-- Group 5: Shiny server reactives --\n")

shiny::testServer(server, {

  # ---- 5a. nothing pressed yet -> empty state ----------------------------
  check("initial result panel shows the empty state",
        grepl("No prediction yet", html_of(output$result_ui)))

  # ---- 5b. valid input -> full result ------------------------------------
  session$setInputs(age = 34, gender = "Female",
                    height = 160, weight = 58, duration = 4,
                    smoking = "No", alcohol = "No",
                    conditions = "None",
                    symptoms = c("fever", "joint_pain", "body_pain",
                                 "headache", "skin_rash"))
  session$setInputs(predict_btn = 1)
  out <- html_of(output$result_ui)

  check("predicted disease is displayed",        grepl("Dengue", out))
  check("confidence percentage is displayed",    grepl("Confidence:", out))
  check("recommended specialist is displayed",   grepl("Specialist", out))
  check("medicine section is displayed",         grepl("Medicine information", out))
  check("home care section is displayed",        grepl("Home care", out))
  check("precautions section are displayed",     grepl("precautions", out, ignore.case = TRUE))
  check("medical disclaimer is displayed",       grepl("medical advice", out, ignore.case = TRUE))
  check("no raw HTML/escape artefacts",          !grepl("&lt;|&amp;lt;", out))

  # ---- v2 additions inside the result ------------------------------------
  check("Top 3 section is displayed",            grepl("Top 3 possible diseases", out))
  check("all three ranks are shown",             grepl("#1", out) && grepl("#3", out))
  check("symptom matching score section shown",  grepl("Symptom matching score", out))
  check("match labels are shown",
        grepl("Typically", out) || grepl("typical", out, ignore.case = TRUE))
  check("disease information card shown",        grepl("Disease information card", out))
  check("common symptoms shown in info card",    grepl("Common symptoms", out))
  check("causes shown in info card",             grepl("Common causes", out))
  check("patient profile summary shown",         grepl("Patient profile recorded", out))
  check("BMI shown in profile summary",          grepl("BMI", out))
  check("BMI band is computed",                  grepl("Normal|Overweight|Underweight|Obese", out))

  # ---- 5c. no symptoms selected -> graceful warning ----------------------
  session$setInputs(symptoms = character(0))   # separate event, as in a browser
  session$setInputs(age = 34, gender = "Female")
  session$setInputs(predict_btn = 2)
  check("no symptoms selected shows a warning, not a crash",
        grepl("No symptoms selected", html_of(output$result_ui)))

  # ---- 5c-2. symptom SEARCH must not lose or invent selections ----------
  # Regression test for a real bug: the visible-symptom set was compared using
  # the DISPLAY LABELS ("Fever") instead of the internal keys ("fever"), so a
  # cleared selection came back on the next prediction.
  session$setInputs(symptoms = c("fever", "cough"))
  check("selecting symptoms registers them",
        setequal(current_symptoms(), c("fever", "cough")))

  session$setInputs(symptom_search = "joint")      # hides fever & cough
  check("search narrows the visible choices",
        length(visible_choices()) < length(SYMPTOM_COLS) &&
        length(visible_choices()) > 0)
  check("search keeps the previously selected symptoms",
        setequal(current_symptoms(), c("fever", "cough")))
  check("searched choices are keyed by symptom key, not label",
        all(unname(visible_choices()) %in% SYMPTOM_COLS))

  session$setInputs(symptoms = c("fever", "cough"))  # tick one more while filtered
  session$setInputs(symptom_search = "")             # clear the search
  check("clearing the search restores all choices",
        length(visible_choices()) == length(SYMPTOM_COLS))
  check("symptoms selected while filtered are kept",
        setequal(current_symptoms(), c("fever", "cough")))

  check("symptom counter reflects the selection",
        grepl("2 symptoms selected", html_of(output$sym_count)))
  check("chips are rendered for the selection",
        grepl("hc-chip", html_of(output$selected_chips)))

  # ---- 5c-3. clearing really clears (the exact bug that was fixed) -------
  session$setInputs(symptoms = character(0))
  check("clearing the checkbox list empties the selection",
        length(current_symptoms()) == 0L)
  check("clearing empties the stored selection too",
        length(selected_syms()) == 0L)
  check("counter returns to zero",
        grepl("0 symptoms selected", html_of(output$sym_count)))

  # ---- 5c-4. restore a valid state for the following checks -------------
  session$setInputs(age = 34, gender = "Female")
  session$setInputs(symptoms = character(0))

  # ---- 5d. invalid age -> graceful warning -------------------------------
  session$setInputs(age = 999, gender = "Male", symptoms = c("fever"))
  session$setInputs(predict_btn = 3)
  check("out-of-range age shows a warning, not a crash",
        grepl("Age must be between 1 and 120", html_of(output$result_ui)))

  # ---- 5e. NA age -> graceful warning ------------------------------------
  session$setInputs(age = NA, gender = "Male", symptoms = c("fever"))
  session$setInputs(predict_btn = 4)
  check("NA age shows a warning, not a crash",
        grepl("missing or not a number", html_of(output$result_ui)))

  # ---- 5f. v2 validation: bad height / weight / duration -----------------
  session$setInputs(age = 30, gender = "Male", height = 20, weight = 60,
                    duration = 3, symptoms = c("fever"))
  session$setInputs(predict_btn = 5)
  check("impossible height is rejected",
        grepl("Height", html_of(output$result_ui)))

  session$setInputs(age = 30, gender = "Male", height = 170, weight = 999,
                    duration = 3, symptoms = c("fever"))
  session$setInputs(predict_btn = 6)
  check("impossible weight is rejected",
        grepl("Weight", html_of(output$result_ui)))

  session$setInputs(age = 30, gender = "Male", height = 170, weight = 70,
                    duration = 5000, symptoms = c("fever"))
  session$setInputs(predict_btn = 7)
  check("impossible symptom duration is rejected",
        grepl("duration", html_of(output$result_ui), ignore.case = TRUE))

  # ---- 5g. several problems reported together ---------------------------
  session$setInputs(symptoms = character(0))   # separate event, as in a browser
  session$setInputs(age = 999, gender = "Male", height = 170, weight = 70,
                    duration = 3)
  session$setInputs(predict_btn = 8)
  multi <- html_of(output$result_ui)
  check("multiple validation problems are listed together",
        grepl("fix the following", multi))

  # ---- 5h. single symptom still works ------------------------------------
  session$setInputs(age = 30, gender = "Male", height = 170, weight = 70,
                    duration = 3, symptoms = c("fever"))
  session$setInputs(predict_btn = 9)
  check("a single symptom still produces a prediction",
        grepl("Predicted disease", html_of(output$result_ui)))

  # ---- 5i. low confidence warning ----------------------------------------
  session$setInputs(age = 50, gender = "Female", symptoms = SYMPTOM_COLS)
  session$setInputs(predict_btn = 10)
  check("all-symptoms case returns a valid result",
        grepl("Predicted disease", html_of(output$result_ui)))

  # ---- 5j. Sample patient fills the form --------------------------------
  session$setInputs(sample_btn = 1)
  check("Load Sample Patient sets symptoms",
        length(session$input$symptoms) > 0)

  # ---- 5k. history accumulates ------------------------------------------
  hist_df <- history_df()
  check("history captured the successful predictions", nrow(hist_df) >= 3)
  check("history has the expected columns",
        all(c("Patient", "Predicted", "Confidence", "Second", "Third",
              "Match", "Specialist", "Urgency") %in% names(hist_df)))
  check("history confidence is stored as a percentage string",
        all(grepl("%$", hist_df$Confidence)))
  check("history statistics render",
        nzchar(html_of(output$hist_total)) &&
        nzchar(html_of(output$hist_top_disease)) &&
        nzchar(html_of(output$hist_avg_conf)))
  check("history table renders once populated",
        grepl("table", html_of(output$history_body)))

  # ---- 5l. downloads are wired up ---------------------------------------
  check("history CSV download handler exists",
        !is.null(output$download_history))
  check("prediction CSV download handler exists",
        !is.null(output$download_prediction))
  check("HTML report download handler exists",
        !is.null(output$download_report))

  # ---- 5m. reset vs new patient -----------------------------------------
  session$setInputs(reset_btn = 1)
  check("Reset Form clears the result panel",
        grepl("No prediction yet", html_of(output$result_ui)))
  check("Reset Form clears the symptom selection",
        length(selected_syms()) == 0L)

  # ---- 5n. analytics outputs exist ---------------------------------------
  check("disease-distribution plot data present",  nrow(DISEASE_DIST) == length(DISEASE_LEVELS))
  check("symptom-frequency data present",          nrow(SYMPTOM_DIST) == length(SYMPTOM_COLS))
  check("per-disease accuracy data present",
        nrow(MODEL_INFO$per_class) == length(DISEASE_LEVELS))
  check("confusion matrix has one row per disease",
        nrow(as.data.frame.matrix(MODEL_INFO$cm_table)) == length(DISEASE_LEVELS))
})

# ============================================================================
# SUMMARY
# ============================================================================
cat("\n============================================================\n")
cat(sprintf(" RESULT: %d passed, %d failed\n", pass, fail))
cat("============================================================\n")
if (fail == 0L) {
  cat(" ALL TESTS PASSED - the application works end to end.\n")
} else {
  cat(" SOME TESTS FAILED - see the FAIL lines above.\n")
}
cat("============================================================\n\n")
