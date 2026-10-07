# ============================================================================
#  HEALTHCARE RECOMMENDATION SYSTEM USING R
#  ---------------------------------------------------------------------------
#  A Shiny web application that takes a patient's basic details and symptoms,
#  predicts the most likely disease using a machine-learning classifier, and
#  shows a dataset-based healthcare recommendation.
#
#  Technology : R + Shiny + dplyr + ggplot2 + randomForest (+ caret if present)
#  Data       : CSV files only (no database, no API, no Python, no JS framework)
#
#  HOW TO RUN
#    1) setwd("path/to/Healthcare-Recommendation")
#    2) shiny::runApp()
#
#  EDUCATIONAL PROJECT - The prediction and the medicine information are for
#  academic demonstration only and are NOT medical advice.
# ============================================================================


# ============================================================================
# SECTION 1 - PACKAGE CHECK  (clear install instructions if anything missing)
# ============================================================================

REQUIRED_PKGS <- c("shiny", "dplyr", "ggplot2")   # must be present
MODEL_PKGS    <- c("randomForest")                # preferred model engine

.installed <- rownames(utils::installed.packages())

.missing_required <- setdiff(REQUIRED_PKGS, .installed)

if (length(.missing_required) > 0L) {
  cat("\n")
  cat("================================================================\n")
  cat("  HEALTHCARE RECOMMENDATION SYSTEM - MISSING R PACKAGES\n")
  cat("================================================================\n")
  cat("  These required packages are NOT installed:\n")
  cat("     ", paste(.missing_required, collapse = ", "), "\n\n")
  cat("  Fix: open the R console and run this ONE command:\n\n")
  cat('     install.packages(c("shiny", "dplyr", "ggplot2", "randomForest"))\n\n')
  cat("  Wait for it to finish (it may take a few minutes the first time),\n")
  cat("  then start the app again with:\n\n")
  cat("     shiny::runApp()\n")
  cat("================================================================\n\n")
  stop("Missing required R packages: ", paste(.missing_required, collapse = ", "),
       call. = FALSE)
}

HAS_RF     <- "randomForest" %in% .installed
HAS_CARET  <- "caret"        %in% .installed

if (!HAS_RF) {
  cat("\n[NOTE] Package 'randomForest' is not installed, so the app will use a\n",
      "       single decision tree (package 'rpart', which ships with R).\n",
      "       For better accuracy install it with: install.packages(\"randomForest\")\n\n",
      sep = "")
}

suppressPackageStartupMessages({
  library(shiny)
  library(dplyr)
  library(ggplot2)
})

set.seed(123)   # reproducible train/test split and reproducible forest


# ============================================================================
# SECTION 2 - CONFIGURATION  (single source of truth for symptom columns)
# ============================================================================
# To use your OWN dataset: keep the same column names as the keys below, or
# rename the keys here. Each symptom column must contain 0 = absent / 1 = present.
# ----------------------------------------------------------------------------

SYMPTOM_LABELS <- c(
  fever                = "Fever",
  cough                = "Cough",
  headache             = "Headache",
  fatigue              = "Fatigue / Weakness",
  nausea               = "Nausea",
  vomiting             = "Vomiting",
  sneezing             = "Sneezing",
  sore_throat          = "Sore Throat",
  body_pain            = "Body Pain",
  stomach_pain         = "Stomach Pain",
  breathing_difficulty = "Breathing Difficulty",
  runny_nose           = "Runny Nose",
  chills               = "Chills / Shivering",
  joint_pain           = "Joint Pain",
  skin_rash            = "Skin Rash",
  chest_pain           = "Chest Pain",
  dizziness            = "Dizziness",
  loss_of_appetite     = "Loss of Appetite",
  diarrhea             = "Diarrhea / Loose Motions",
  constipation         = "Constipation",
  burning_urination    = "Burning Urination",
  blurred_vision       = "Blurred Vision",
  night_sweats         = "Night Sweats",
  weight_loss          = "Weight Loss",
  yellow_eyes          = "Yellow Eyes / Skin",
  palpitations         = "Palpitations",
  frequent_urination   = "Frequent Urination",
  excessive_thirst     = "Excessive Thirst"
)

SYMPTOM_COLS  <- names(SYMPTOM_LABELS)                  # 28 binary columns
MODEL_VARS    <- c(SYMPTOM_COLS, "Age", "Gender")       # model predictors
TARGET_COL    <- "Disease"                              # model target

MODEL_FORMULA <- stats::as.formula(
  paste(TARGET_COL, "~", paste(MODEL_VARS, collapse = " + "))
)

# An independent symptom set used together with the chosen symptoms, so that
# the forest always sees the same predictors it was trained on.
SEED_SYMPTOM <- "fatigue"

# Colour palette (healthcare theme)
COL_PRIMARY <- "#0b7fa8"
COL_DARK    <- "#0b5c85"
COL_TEAL    <- "#17a2b8"
COL_GREEN   <- "#1e9e6a"
COL_AMBER   <- "#d97706"
COL_RED     <- "#d64545"
COL_GREY    <- "#5b7186"

# --- Patient profile options (v2 feature 1: basic health details) ----------
SMOKING_CHOICES  <- c("No", "Yes")
ALCOHOL_CHOICES  <- c("No", "Yes")
CONDITION_CHOICES <- c(
  "Diabetes"         = "Diabetes",
  "High Blood Pressure" = "Hypertension",
  "Asthma"           = "Asthma",
  "Heart Disease"    = "Heart Disease",
  "None"             = "None"
)

# A symptom is treated as "typical" of a disease when at least this share of
# the patients having that disease in the dataset also report the symptom.
MATCH_TYPICAL_CUTOFF <- 0.50
MATCH_PARTIAL_CUTOFF <- 0.20

# Small null-coalescing helper (base R 4.4+ has %||%, this keeps us compatible)
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0L) b else a


# ============================================================================
# SECTION 3 - LOAD THE CSV DATA
# ============================================================================
# The working directory is the app folder when you use shiny::runApp(), but a
# few fallback locations are searched so the app also works if it is launched
# from a parent folder.
# ----------------------------------------------------------------------------

find_data_file <- function(fname) {
  candidates <- c(
    fname,
    file.path("data", fname),
    file.path("Healthcare-Recommendation", fname),
    file.path("..", fname),
    file.path("..", "..", fname)
  )
  hit <- candidates[file.exists(candidates)]
  if (length(hit) == 0L) {
    # last resort: look one/two levels down
    deep <- list.files(".", pattern = paste0("^", fname, "$"),
                       recursive = TRUE, full.names = TRUE)
    if (length(deep) > 0L) hit <- deep[1]
  }
  if (length(hit) == 0L) return(NA_character_)
  normalizePath(hit[1], mustWork = FALSE)
}

read_csv_safe <- function(fname, label) {
  path <- find_data_file(fname)
  if (is.na(path)) {
    stop(sprintf(
      "\n\n[FILE NOT FOUND] %s\n  '%s' could not be found.\n  Make sure it sits in the SAME folder as app.R, then restart the app.\n",
      label, fname), call. = FALSE)
  }
  df <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  if (nrow(df) == 0L) stop(sprintf("[EMPTY FILE] %s contains no rows.", label),
                           call. = FALSE)
  attr(df, "source_path") <- path
  df
}

HEALTHCARE_RAW <- read_csv_safe("healthcare_data.csv", "Healthcare dataset")
RECOMMENDATIONS <- read_csv_safe("recommendations.csv",  "Recommendation table")


# ============================================================================
# SECTION 4 - DATA CLEANING + PREPROCESSING
# ============================================================================

#' Clean the raw healthcare dataset:
#'  * drop records with no disease label
#'  * force every symptom column to integer 0/1 and replace NA with 0
#'  * repair Age (NA / out-of-range -> median age)
#'  * repair Gender (NA / unexpected -> most common value)
#'  * make Disease a factor (required for CLASSIFICATION, not regression)
clean_healthcare_data <- function(df) {

  # --- 4.1 keep only rows that have a disease label ------------------------
  if (!TARGET_COL %in% names(df)) {
    stop(sprintf("Column '%s' is missing from healthcare_data.csv.", TARGET_COL),
         call. = FALSE)
  }
  df <- df[!is.na(df[[TARGET_COL]]) & trimws(as.character(df[[TARGET_COL]])) != "", ,
           drop = FALSE]

  # --- 4.2 symptom columns -> integer 0/1 -----------------------------------
  for (s in SYMPTOM_COLS) {
    if (!s %in% names(df)) {
      df[[s]] <- 0L                      # column absent in a custom dataset
      next
    }
    v <- suppressWarnings(as.numeric(df[[s]]))   # "yes"/"no" becomes NA -> 0
    v[is.na(v)] <- 0
    df[[s]] <- as.integer(v > 0)
  }

  # --- 4.3 Age ---------------------------------------------------------------
  age <- suppressWarnings(as.numeric(df$Age))
  if (is.null(age)) age <- rep(NA_real_, nrow(df))
  med_age <- stats::median(age, na.rm = TRUE)
  if (!is.finite(med_age)) med_age <- 30
  age[!is.finite(age) | age < 1 | age > 120] <- med_age
  df$Age <- round(age)

  # --- 4.4 Gender ------------------------------------------------------------
  g <- trimws(as.character(df$Gender))
  g[is.na(g) | !g %in% c("Male", "Female")] <- "Male"
  df$Gender <- factor(g, levels = c("Male", "Female"))

  # --- 4.5 target as factor --------------------------------------------------
  df[[TARGET_COL]] <- factor(trimws(as.character(df[[TARGET_COL]])))

  # --- 4.6 drop diseases that have no recommendation row ---------------------
  df
}

HEALTHCARE_DATA <- clean_healthcare_data(HEALTHCARE_RAW)
HEALTHCARE_DATA <- HEALTHCARE_DATA[!duplicated(HEALTHCARE_DATA), , drop = FALSE]
rownames(HEALTHCARE_DATA) <- NULL

DISEASE_LEVELS <- levels(HEALTHCARE_DATA[[TARGET_COL]])

# Keep only recommendation rows for diseases the model can actually predict
RECOMMENDATIONS <- RECOMMENDATIONS[
  as.character(RECOMMENDATIONS[[TARGET_COL]]) %in% DISEASE_LEVELS, , drop = FALSE]
RECOMMENDATIONS[[TARGET_COL]] <- factor(as.character(RECOMMENDATIONS[[TARGET_COL]]),
                                        levels = DISEASE_LEVELS)

#' Build the model matrix (29 predictors) from a data frame of raw inputs.
prepare_features <- function(df, default_age = 30) {
  out <- as.data.frame(
    matrix(0L, nrow = nrow(df), ncol = length(SYMPTOM_COLS),
           dimnames = list(NULL, SYMPTOM_COLS)),
    stringsAsFactors = FALSE
  )
  for (s in SYMPTOM_COLS) {
    if (s %in% names(df)) {
      v <- suppressWarnings(as.numeric(df[[s]]))
      v[is.na(v)] <- 0
      out[[s]] <- as.integer(v > 0)
    }
  }
  age <- suppressWarnings(as.numeric(df$Age))
  age[!is.finite(age)] <- default_age
  out$Age <- age

  g <- as.character(df$Gender)
  g[is.na(g) | !g %in% c("Male", "Female")] <- "Male"
  out$Gender <- factor(g, levels = c("Male", "Female"))

  out[, MODEL_VARS, drop = FALSE]      # exact same columns / order as training
}


# ============================================================================
# SECTION 5 - TRAIN THE MACHINE-LEARNING MODEL
# ============================================================================

train_healthcare_model <- function(data) {

  engine <- if (HAS_RF) "randomForest" else "rpart"

  # --- 5.1 stratified 75 / 25 train-test split ------------------------------
  set.seed(123)
  train_idx <- stratified_split(data[[TARGET_COL]], p = 0.75)
  train_df  <- data[ train_idx, , drop = FALSE]
  test_df   <- data[-train_idx, , drop = FALSE]

  # --- 5.2 fit ---------------------------------------------------------------
  model <- if (engine == "randomForest") {
    randomForest::randomForest(MODEL_FORMULA, data = train_df,
                               ntree = 300, importance = TRUE)
  } else {
    rpart::rpart(MODEL_FORMULA, data = train_df, method = "class")
  }

  # --- 5.3 evaluate ----------------------------------------------------------
  pred_fun <- function(d) {
    p <- predict(model, newdata = d, type = "class")
    factor(as.character(p), levels = levels(data[[TARGET_COL]]))
  }
  actual    <- factor(as.character(test_df[[TARGET_COL]]),
                      levels = levels(data[[TARGET_COL]]))
  pred_test <- pred_fun(test_df)
  pred_train <- pred_fun(train_df)
  actual_train <- factor(as.character(train_df[[TARGET_COL]]),
                         levels = levels(data[[TARGET_COL]]))

  test_acc  <- mean(pred_test  == actual,       na.rm = TRUE)
  train_acc <- mean(pred_train == actual_train, na.rm = TRUE)

  # --- 5.4 confusion matrix --------------------------------------------------
  cm_table <- table(Predicted = pred_test, Actual = actual)

  cm_stats <- NULL
  if (HAS_CARET) {
    cm_stats <- tryCatch(
      caret::confusionMatrix(pred_test, actual),
      error = function(e) NULL)
  }

  # --- 5.5 per-disease accuracy (for the analytics chart) --------------------
  per_class <- data.frame(
    Disease  = levels(actual),
    Records  = as.integer(table(actual)),
    Accuracy = vapply(levels(actual), function(cl) {
      idx <- which(actual == cl)
      if (length(idx) == 0L) return(NA_real_)
      mean(pred_test[idx] == cl, na.rm = TRUE)
    }, numeric(1)),
    stringsAsFactors = FALSE
  )
  per_class <- per_class[order(per_class$Accuracy), , drop = FALSE]
  rownames(per_class) <- NULL

  # --- 5.6 variable importance ----------------------------------------------
  importance <- NULL
  if (engine == "randomForest") {
    imp <- randomForest::importance(model)
    importance <- data.frame(
      Feature    = rownames(imp),
      Importance = round(as.numeric(imp[, ncol(imp)]), 2),
      stringsAsFactors = FALSE
    )
    importance <- importance[order(-importance$Importance), , drop = FALSE]
    rownames(importance) <- NULL
  }

  list(
    engine       = engine,
    model        = model,
    formula      = MODEL_FORMULA,
    n_train      = nrow(train_df),
    n_test       = nrow(test_df),
    n_diseases   = length(levels(data[[TARGET_COL]])),
    test_acc     = test_acc,
    train_acc    = train_acc,
    cm_table     = cm_table,
    cm_stats     = cm_stats,
    per_class    = per_class,
    importance   = importance,
    trained_ok   = TRUE,
    message      = NULL
  )
}

#' Stratified 75/25 train-test split - keeps every disease represented in both
#' parts, which matters because the classes are not equally frequent.
#' Uses caret::createDataPartition when caret is installed (the standard
#' approach), otherwise falls back to exactly the same logic written in base R.
#' Always returns integer row indices of the TRAINING set.
stratified_split <- function(y, p = 0.75) {

  n <- length(y)

  if (HAS_CARET) {
    res <- tryCatch(caret::createDataPartition(y, p = p, list = FALSE),
                    error = function(e) NULL)
    if (!is.null(res)) {
      idx <- as.integer(res)
      if (length(idx) > 0L) return(idx)
    }
  }

  # ---- base-R fallback: sample 75% of every class separately ---------------
  groups <- split(seq_len(n), as.character(y))
  idx <- unlist(lapply(groups, function(g) {
    k <- max(1L, floor(p * length(g)))
    sample(g, size = k)
  }), use.names = FALSE)

  sort(as.integer(idx))
}

# --- train once, at startup --------------------------------------------------
MODEL_INFO <- tryCatch(
  train_healthcare_model(HEALTHCARE_DATA),
  error = function(e) list(
    engine = if (HAS_RF) "randomForest" else "rpart", model = NULL,
    formula = MODEL_FORMULA, n_train = NA_integer_, n_test = NA_integer_,
    n_diseases = length(DISEASE_LEVELS), test_acc = NA_real_,
    train_acc = NA_real_, cm_table = NULL, cm_stats = NULL,
    per_class = data.frame(Disease = character(), Records = integer(),
                           Accuracy = numeric()),
    importance = NULL, trained_ok = FALSE,
    message = conditionMessage(e))
)

cat("\n----------------------------------------------------------------\n")
cat(" Healthcare Recommendation System - startup summary\n")
cat("----------------------------------------------------------------\n")
cat(" Patient records      :", nrow(HEALTHCARE_DATA), "\n")
cat(" Diseases             :", length(DISEASE_LEVELS), "\n")
cat(" Symptoms tracked     :", length(SYMPTOM_COLS), "\n")
if (isTRUE(MODEL_INFO$trained_ok)) {
  cat(" Model engine         :", MODEL_INFO$engine, "\n")
  cat(" Training records     :", MODEL_INFO$n_train, "\n")
  cat(" Testing records      :", MODEL_INFO$n_test, "\n")
  cat(" Test accuracy        :", sprintf("%.2f%%", 100 * MODEL_INFO$test_acc), "\n")
  cat(" Training accuracy    :", sprintf("%.2f%%", 100 * MODEL_INFO$train_acc), "\n")
} else {
  cat(" Model training FAILED:", MODEL_INFO$message, "\n")
}
cat("----------------------------------------------------------------\n\n")


# ============================================================================
# SECTION 6 - PREDICTION + RECOMMENDATION HELPERS
# ============================================================================

#' Predict the disease for one patient.
#' Returns a list with the top disease, its confidence and the top-3 ranking.
predict_patient <- function(model_info, features) {
  if (!isTRUE(model_info$trained_ok) || is.null(model_info$model)) {
    stop("The prediction model could not be trained, so no prediction is available.")
  }
  if (nrow(features) != 1L) stop("Exactly one patient record is required.")

  prob <- predict(model_info$model, newdata = features, type = "prob")

  # rpart returns a 1-row matrix / vector - normalise both shapes
  prob_vec <- as.numeric(prob[1, ])
  names(prob_vec) <- colnames(prob)
  prob_vec[!is.finite(prob_vec)] <- 0

  if (sum(prob_vec) <= 0) {                     # no probabilities at all
    prob_vec[] <- 0
    prob_vec[[1]] <- 1
  }

  ord   <- order(prob_vec, decreasing = TRUE)
  top_n <- min(3L, length(ord))

  list(
    disease    = names(prob_vec)[ord[1]],
    confidence = as.numeric(prob_vec[ord[1]]),
    ranking    = data.frame(
      Disease     = names(prob_vec)[ord[seq_len(top_n)]],
      Probability = as.numeric(prob_vec[ord[seq_len(top_n)]]),
      stringsAsFactors = FALSE
    )
  )
}

#' Look up the recommendation row for a disease, with a safe fallback.
get_recommendation <- function(disease) {
  row <- RECOMMENDATIONS[as.character(RECOMMENDATIONS[[TARGET_COL]]) == disease, ,
                         drop = FALSE]
  if (nrow(row) == 0L) {
    return(list(
      Disease   = disease,
      Specialist = "General Physician",
      Description = "No extra information is stored in the recommendation file for this disease.",
      Medicines_Educational = "Not available in the dataset.",
      Home_Care = "Take rest and drink plenty of fluids.",
      Precautions = "Please consult a qualified doctor for proper evaluation.",
      Urgency_Level = "Consult a doctor"
    ))
  }
  as.list(row[1, , drop = FALSE])
}

#' Small safe getter so a missing/renamed column never crashes the app.
rec_field <- function(rec, name, fallback = "Not available in the dataset.") {
  v <- rec[[name]]
  if (is.null(v) || length(v) == 0L || is.na(v) || !nzchar(trimws(as.character(v)))) {
    return(fallback)
  }
  trimws(as.character(v))
}

# ---------------------------------------------------------------------------
# 6b. SYMPTOM-TO-DISEASE MATCHING  (v2 feature 7)
# ---------------------------------------------------------------------------
# SYMPTOM_PROFILE is a disease x symptom matrix of P(symptom | disease), i.e.
# "what share of the patients who HAVE this disease also report this symptom".
# It is computed once from healthcare_data.csv, so the matching score is
# evidence from the dataset rather than a hand-written rule.

#' Compute that matrix from the training data.
build_symptom_profile <- function(data) {
  sp <- data %>%
    group_by(.data[[TARGET_COL]]) %>%
    summarise(dplyr::across(dplyr::all_of(SYMPTOM_COLS), mean), .groups = "drop")

  m <- as.matrix(sp[, SYMPTOM_COLS, drop = FALSE])
  rownames(m) <- as.character(sp[[TARGET_COL]])
  m[!is.finite(m)] <- 0
  m
}

#' Match one disease against the symptoms the user selected.
#' Returns the overall match score (0-1) plus a per-symptom breakdown.
match_disease_to_symptoms <- function(disease, selected_syms) {
  empty <- list(
    score = NA_real_, matched = 0L, partial = 0L, unmatched = 0L,
    detail = data.frame(Symptom = character(), Typicality = numeric(),
                        Level = character(), stringsAsFactors = FALSE)
  )
  if (length(selected_syms) == 0L) return(empty)
  if (!disease %in% rownames(SYMPTOM_PROFILE)) return(empty)

  p <- as.numeric(SYMPTOM_PROFILE[disease, selected_syms])
  p[!is.finite(p)] <- 0

  level <- ifelse(p >= MATCH_TYPICAL_CUTOFF, "Typical",
           ifelse(p >= MATCH_PARTIAL_CUTOFF, "Sometimes seen", "Not typical"))

  detail <- data.frame(
    Symptom    = unname(SYMPTOM_LABELS[selected_syms]),
    Typicality = p,
    Level      = level,
    stringsAsFactors = FALSE
  )
  detail <- detail[order(-detail$Typicality), , drop = FALSE]
  rownames(detail) <- NULL

  list(
    score     = mean(p),                     # average typicality of the symptoms
    matched   = sum(level == "Typical"),
    partial   = sum(level == "Sometimes seen"),
    unmatched = sum(level == "Not typical"),
    detail    = detail
  )
}

#' A short readable sentence describing the match, used above the table.
match_sentence <- function(m, n_symptoms) {
  if (n_symptoms == 0L) return("No symptoms were selected.")
  matched <- min(m$matched, n_symptoms)   # never claim more than was given
  sprintf("%d of your %d selected symptom%s %s typical of this disease.",
          matched, n_symptoms, if (n_symptoms == 1L) "" else "s",
          if (matched == 1L) "is" else "are")
}

#' BMI calculation and WHO-style band (educational information only).
bmi_value <- function(height_cm, weight_kg) {
  h <- suppressWarnings(as.numeric(height_cm))
  w <- suppressWarnings(as.numeric(weight_kg))
  if (!is.finite(h) || !is.finite(w) || h <= 0 || w <= 0) return(NA_real_)
  round(w / ((h / 100)^2), 1)
}

bmi_band <- function(bmi) {
  if (is.na(bmi))           return(list(label = "Not available", colour = COL_GREY))
  if (bmi < 18.5)           return(list(label = "Underweight",    colour = COL_AMBER))
  if (bmi < 25)             return(list(label = "Normal",         colour = COL_GREEN))
  if (bmi < 30)             return(list(label = "Overweight",     colour = COL_AMBER))
  list(label = "Obese", colour = COL_RED)
}

confidence_colour <- function(p) {
  if (is.na(p))    return(COL_GREY)
  if (p >= 0.70)   return(COL_GREEN)
  if (p >= 0.40)   return(COL_AMBER)
  COL_RED
}

#' Turn a "A; B; C" field into an HTML bullet list.
bullet_list <- function(text) {
  if (is.null(text) || is.na(text) || !nzchar(trimws(text))) {
    return(tags$p(class = "hc-muted", "Not available in the dataset."))
  }
  items <- trimws(unlist(strsplit(as.character(text), ";")))
  items <- items[nzchar(items)]
  tags$ul(class = "hc-list", lapply(items, function(x)
    tags$li(htmltools::htmlEscape(x))))
}

#' Escape text for safe embedding in HTML (used by the downloadable report).
esc <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- gsub("&", "&amp;", x, fixed = TRUE)
  x <- gsub("<", "&lt;",  x, fixed = TRUE)
  x <- gsub(">", "&gt;",  x, fixed = TRUE)
  x
}

#' Turn "A; B; C" into an HTML bullet list for the downloadable report.
report_bullets <- function(text) {
  if (is.null(text) || is.na(text) || !nzchar(trimws(text))) return("<p>-</p>")
  items <- trimws(unlist(strsplit(as.character(text), ";")))
  items <- items[nzchar(items)]
  if (length(items) == 0L) return("<p>-</p>")
  paste0("<ul>", paste0("<li>", esc(items), "</li>", collapse = ""), "</ul>")
}


# ============================================================================
# SECTION 7 - PRE-COMPUTED ANALYTICS DATA
# ============================================================================

DISEASE_DIST <- HEALTHCARE_DATA %>%
  count(.data[[TARGET_COL]], name = "Records") %>%
  arrange(desc(.data$Records)) %>%
  mutate(Disease = as.character(.data[[TARGET_COL]]))

SYMPTOM_DIST <- data.frame(
  Symptom = unname(SYMPTOM_LABELS[SYMPTOM_COLS]),
  Key     = SYMPTOM_COLS,
  Count   = as.integer(colSums(HEALTHCARE_DATA[, SYMPTOM_COLS, drop = FALSE])),
  stringsAsFactors = FALSE
)
SYMPTOM_DIST$Percent <- 100 * SYMPTOM_DIST$Count / nrow(HEALTHCARE_DATA)
SYMPTOM_DIST <- SYMPTOM_DIST[order(-SYMPTOM_DIST$Count), , drop = FALSE]
rownames(SYMPTOM_DIST) <- NULL

# P(symptom | disease) matrix - the evidence behind the symptom-matching score
SYMPTOM_PROFILE <- build_symptom_profile(HEALTHCARE_DATA)

# Number of patients per disease, keyed by disease name (used in the info card)
DISEASE_COUNTS <- stats::setNames(DISEASE_DIST$Records, DISEASE_DIST$Disease)


#' Build the downloadable HTML report for one prediction result.
#' Kept as a plain function (not inline in the download handler) so that it
#' can be unit-tested without a browser.
build_report_html <- function(r) {
  if (is.null(r) || !isTRUE(r$ok)) {
    return(paste0(
  "<!DOCTYPE html><html><head><meta charset='utf-8'>",
  "<title>Healthcare Report</title></head><body>",
  "<h1>Healthcare Recommendation System</h1>",
  "<p>No successful prediction is currently displayed, so there is ",
  "nothing to report. Run a prediction first, then generate the report again.</p>",
  "</body></html>"))
  }

  rec <- r$rec
  conf <- 100 * r$confidence
  prof <- rec_field(rec, "Specialist", "General Physician")

  rows2 <- ""
  if (nrow(r$ranking) > 1L) {
    rows2 <- paste0(vapply(seq_len(nrow(r$ranking)), function(i) {
      sprintf("<tr><td>%d</td><td>%s</td><td>%.1f%%</td></tr>",
              i, esc(r$ranking$Disease[i]), 100 * r$ranking$Probability[i])
    }, character(1)), collapse = "")
  }

  match_rows <- paste0(vapply(seq_len(nrow(r$match$detail)), function(i) {
    sprintf("<tr><td>%s</td><td>%.0f%%</td><td>%s</td></tr>",
            esc(r$match$detail$Symptom[i]),
            100 * r$match$detail$Typicality[i],
            esc(r$match$detail$Level[i]))
  }, character(1)), collapse = "")

  html <- paste0(
'<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Healthcare Report - ', esc(r$disease), '</title>
<style>
  body { font-family: "Segoe UI", Roboto, Arial, sans-serif; color:#22303f;
     max-width: 860px; margin: 26px auto; padding: 0 20px; line-height:1.55; }
  h1 { color:#0b5c85; font-size:25px; margin-bottom:2px; }
  .sub { color:#6b8299; font-size:13px; margin-bottom:22px; }
  h2 { color:#0b5c85; font-size:17px; border-bottom:2px solid #e1ecf4;
   padding-bottom:5px; margin-top:26px; }
  table { border-collapse: collapse; width:100%; font-size:13.5px; margin-top:8px; }
  th, td { border:1px solid #dbe9f0; padding:6px 9px; text-align:left; }
  th { background:#f2f8fc; color:#0b5c85; }
  .big { font-size:29px; font-weight:700; color:#10513a; }
  .pill { display:inline-block; background:#eaf3f8; color:#0b5c85;
      border-radius:14px; padding:3px 11px; font-size:12.5px; font-weight:700; }
  .warn { background:#fff9ec; border:1px solid #f2dfb6; border-left:5px solid #d97706;
      padding:12px 15px; border-radius:8px; font-size:13.5px; color:#78450a;
      margin-top:18px; }
  .meta { font-size:12.5px; color:#6b8299; margin-top:30px;
      border-top:1px dashed #dbe9f0; padding-top:10px; }
  @media print { body { margin:0; } h2 { page-break-after: avoid; }
             table { page-break-inside: avoid; } }
</style></head><body>

<h1>Healthcare Recommendation System</h1>
<div class="sub">Prediction report generated by a college R / Shiny project &middot; ',
esc(r$time), '</div>

<h2>Prediction</h2>
<div class="big">', esc(r$disease), '</div>
<p>Confidence: <b>', sprintf("%.1f%%", conf), '</b> &nbsp; ',
   '<span class="pill">Match score: ',
   if (is.na(r$match$score)) "-" else sprintf("%.0f%%", 100 * r$match$score),
   '</span></p>

<h2>Patient profile</h2>
<table>
  <tr><th>Age</th><td>', esc(sprintf("%d years", r$age)), '</td>
  <th>Gender</th><td>', esc(r$gender), '</td></tr>
  <tr><th>Height</th><td>', esc(sprintf("%.0f cm", r$height)), '</td>
  <th>Weight</th><td>', esc(sprintf("%.0f kg", r$weight)), '</td></tr>
  <tr><th>BMI</th><td>',
  if (is.na(r$bmi)) "-" else esc(sprintf("%.1f (%s)", r$bmi, bmi_band(r$bmi)$label)),
  '</td><th>Symptom duration</th><td>',
  esc(sprintf("%.0f days", r$duration)), '</td></tr>
  <tr><th>Smoking</th><td>', esc(r$smoking), '</td>
  <th>Alcohol</th><td>', esc(r$alcohol), '</td></tr>
  <tr><th>Existing conditions</th><td colspan="3">',
  if (length(r$conditions) == 0L) "None reported"
  else esc(paste(r$conditions, collapse = ", ")), '</td></tr>
</table>

<h2>Top 3 possible diseases</h2>
<table>
  <tr><th>Rank</th><th>Disease</th><th>Model probability</th></tr>',
  rows2,
'</table>

<h2>Symptoms given</h2>
<p>', esc(paste(unname(SYMPTOM_LABELS[r$symptoms]), collapse = ", ")), '</p>

<h2>Symptom matching score</h2>
<p>', esc(match_sentence(r$match, r$n_symptoms)), '</p>
<table>
  <tr><th>Symptom</th><th>Patients with this disease showing it</th><th>Match</th></tr>',
  match_rows,
'</table>

<h2>Healthcare recommendation</h2>
<p><b>Recommended specialist:</b> ', esc(prof), '<br>
   <b>Action advised:</b> ', esc(rec_field(rec, "Urgency_Level", "Consult a doctor")), '</p>
<p>', esc(rec_field(rec, "Description")), '</p>

<h2>Common symptoms of this disease</h2>', report_bullets(rec_field(rec, "Common_Symptoms")),
'<h2>Common causes</h2>', report_bullets(rec_field(rec, "Causes")),
'<h2>Home care</h2>', report_bullets(rec_field(rec, "Home_Care")),
'<h2>When to see a doctor / precautions</h2>', report_bullets(rec_field(rec, "Precautions")),

'<h2>Medicine information (educational - from the dataset)</h2>
<div class="warn"><b>Not medical advice.</b> This information is general,
publicly documented material stored in recommendations.csv for academic
demonstration. It is not a prescription and does not replace a doctor.
Never self-medicate.</div>', report_bullets(rec_field(rec, "Medicines_Educational")),

'<div class="warn"><b>Medical disclaimer.</b> This report was produced by a
student project using a statistical model and a stored dataset. It is not a
medical diagnosis or professional advice. Always consult a qualified doctor
for real health problems. In an emergency, go to the nearest hospital.</div>

<div class="meta">Model: ', esc(if (isTRUE(MODEL_INFO$trained_ok)) MODEL_INFO$engine else "unavailable"),
' &nbsp;|&nbsp; Test accuracy: ',
esc(if (isTRUE(MODEL_INFO$trained_ok)) sprintf("%.2f%%", 100 * MODEL_INFO$test_acc) else "-"),
' &nbsp;|&nbsp; Dataset: ', esc(format(nrow(HEALTHCARE_DATA), big.mark = ",")),
' records, ', esc(length(DISEASE_LEVELS)), ' diseases<br>
To save as PDF: open this file in a browser and choose Print &rarr; Save as PDF.</div>

</body></html>')
  html
}

# ============================================================================
# SECTION 8 - USER INTERFACE
# ============================================================================

APP_CSS <- "
/* ---------- page ---------- */
body {
  background-color: #f2f7fb;
  font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  color: #22303f;
}
.container-fluid { max-width: 1400px; padding-left: 22px; padding-right: 22px; }

/* ---------- top navigation (covers Bootstrap 3 and 5 markup) ---------- */
.navbar, .navbar-default {
  background: linear-gradient(90deg, #0b5c85 0%, #0d8ba6 100%);
  border: none; border-radius: 0; margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(11, 92, 133, .25);
}
.navbar-brand, .navbar-header .navbar-brand {
  color: #ffffff !important; font-weight: 700; font-size: 18px;
  letter-spacing: .2px; height: auto; padding: 15px 15px;
}
.navbar-brand span.hc-cross { color: #7fe3f0; }
.navbar-nav > li > a, .navbar .navbar-nav .nav-link {
  color: rgba(255,255,255,.88) !important; font-weight: 500;
  font-size: 14.5px; transition: background .15s ease;
}
.navbar-nav > li > a:hover, .navbar .navbar-nav .nav-link:hover {
  background: rgba(255,255,255,.14) !important; color: #fff !important;
}
.navbar-nav > .active > a, .navbar-nav > .active > a:hover,
.navbar .navbar-nav .nav-link.active {
  background: rgba(255,255,255,.22) !important; color: #fff !important;
  font-weight: 600;
}

/* ---------- cards ---------- */
.hc-card {
  background: #ffffff; border: 1px solid #e1ecf4; border-radius: 12px;
  padding: 18px 20px; margin-bottom: 18px;
  box-shadow: 0 2px 8px rgba(13, 74, 110, .06);
}
.hc-card h3, .hc-card h4 { margin-top: 0; color: #0b5c85; font-weight: 700; }
.hc-card h3 { font-size: 19px; }
.hc-card h4 { font-size: 16px; }
.hc-hero {
  background: linear-gradient(120deg, #0b5c85 0%, #1195b3 100%);
  color: #fff; border-radius: 12px; padding: 26px 28px; margin-bottom: 20px;
  box-shadow: 0 4px 14px rgba(11, 92, 133, .22);
}
.hc-hero h1 { margin: 0 0 8px 0; font-size: 27px; font-weight: 700; color: #fff; }
.hc-hero p  { margin: 0; font-size: 15px; line-height: 1.55;
              color: rgba(255,255,255,.93); max-width: 900px; }

/* ---------- stat cards ---------- */
.hc-stat {
  background: #fff; border: 1px solid #e1ecf4; border-left: 5px solid #0b7fa8;
  border-radius: 10px; padding: 15px 18px; margin-bottom: 16px; height: 100%;
  box-shadow: 0 2px 8px rgba(13, 74, 110, .06);
}
.hc-stat .hc-stat-value {
  font-size: 27px; font-weight: 700; color: #0b5c85; line-height: 1.15;
}
.hc-stat .hc-stat-label {
  font-size: 12px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .7px; color: #6b8299; margin-top: 5px;
}
.hc-stat.hc-green { border-left-color: #1e9e6a; }
.hc-stat.hc-green .hc-stat-value { color: #15764f; }
.hc-stat.hc-teal  { border-left-color: #17a2b8; }
.hc-stat.hc-teal  .hc-stat-value { color: #10798a; }

/* ---------- workflow steps ---------- */
.hc-flow { display: flex; flex-wrap: wrap; gap: 12px; }
.hc-step {
  flex: 1 1 180px; background: #f7fbfd; border: 1px dashed #bcd8e7;
  border-radius: 10px; padding: 14px 15px;
}
.hc-step .hc-step-no {
  display: inline-block; width: 26px; height: 26px; line-height: 26px;
  text-align: center; border-radius: 50%; background: #0b7fa8; color: #fff;
  font-weight: 700; font-size: 13px; margin-bottom: 8px;
}
.hc-step .hc-step-title { font-weight: 700; color: #0b5c85; font-size: 14.5px;
                          margin-bottom: 3px; }
.hc-step .hc-step-text  { font-size: 13px; color: #5b7186; line-height: 1.45; }

/* ---------- inputs ---------- */
.form-group label, .control-label { font-weight: 600; color: #3d566e;
                                    font-size: 13.5px; }
.form-control {
  border: 1px solid #cfe0ec; border-radius: 8px; height: 38px;
  box-shadow: none; color: #22303f;
}
.form-control:focus { border-color: #0b7fa8; box-shadow: 0 0 0 3px rgba(11,127,168,.13); }
.well, .sidebar-panel {
  background: #fff; border: 1px solid #e1ecf4; border-radius: 12px;
  box-shadow: 0 2px 8px rgba(13,74,110,.06);
}

/* ---------- symptom checkbox grid ---------- */
#symptoms .shiny-options-group {
  display: grid; grid-template-columns: 1fr 1fr; gap: 2px 10px;
}
.hc-symptom-box {
  max-height: 330px; overflow-y: auto; border: 1px solid #e1ecf4;
  border-radius: 9px; padding: 10px 12px; background: #fbfdff;
}
.hc-symptom-box .checkbox, .hc-symptom-box .form-check { margin-top: 3px; margin-bottom: 3px; }
.hc-symptom-box label { font-weight: 500; font-size: 13px; color: #3d566e; }
.hc-symptom-box input[type=checkbox] { margin-top: 3px; }

/* ---------- result ---------- */
.hc-result {
  background: linear-gradient(120deg, #eaf7f1 0%, #eef8fc 100%);
  border: 1px solid #bfe6d5; border-radius: 12px; padding: 20px 22px;
  margin-bottom: 18px;
}
.hc-result .hc-result-label {
  font-size: 12px; font-weight: 700; letter-spacing: 1px;
  text-transform: uppercase; color: #15764f;
}
.hc-result .hc-result-name {
  font-size: 30px; font-weight: 700; color: #10513a; line-height: 1.2;
  margin: 4px 0 10px 0;
}
.hc-bar-wrap {
  background: #dbe9f0; border-radius: 20px; height: 13px; overflow: hidden;
  width: 100%; max-width: 460px;
}
.hc-bar { height: 100%; border-radius: 20px; }
.hc-bar-text { font-size: 13px; color: #3d566e; margin-top: 6px; font-weight: 600; }

.hc-alt-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 0; border-bottom: 1px dashed #dbe9f0; font-size: 13.5px;
}
.hc-alt-row:last-child { border-bottom: none; }
.hc-alt-name { color: #3d566e; }
.hc-alt-pct  { font-weight: 700; color: #0b7fa8; }

/* ---------- badges ---------- */
.hc-badge {
  display: inline-block; padding: 4px 11px; border-radius: 20px;
  font-size: 12px; font-weight: 700; letter-spacing: .3px;
  background: #eaf3f8; color: #0b5c85;
}
.hc-badge-red   { background: #fdeaea; color: #b33030; }
.hc-badge-amber { background: #fdf3e2; color: #a15c05; }
.hc-badge-green { background: #e7f7ef; color: #15764f; }

/* ---------- lists ---------- */
ul.hc-list { padding-left: 20px; margin: 6px 0 0 0; }
ul.hc-list li { font-size: 14px; line-height: 1.6; color: #34495e;
                margin-bottom: 3px; }

/* ---------- notes and alerts ---------- */
.hc-note {
  background: #fff9ec; border: 1px solid #f2dfb6; border-left: 5px solid #d97706;
  border-radius: 9px; padding: 13px 16px; font-size: 13.5px; color: #78450a;
  line-height: 1.55; margin-bottom: 16px;
}
.hc-note strong { color: #6b3d05; }
.hc-info {
  background: #eef7fc; border: 1px solid #c9e3f1; border-left: 5px solid #0b7fa8;
  border-radius: 9px; padding: 13px 16px; font-size: 13.5px; color: #23506b;
  line-height: 1.55; margin-bottom: 16px;
}
.hc-empty {
  background: #f8fbfd; border: 2px dashed #cfe0ec; border-radius: 12px;
  padding: 46px 24px; text-align: center; color: #7590a6; margin-bottom: 18px;
}
.hc-empty .hc-empty-icon { font-size: 40px; color: #a9c4d6; margin-bottom: 10px; }
.hc-empty .hc-empty-title { font-size: 17px; font-weight: 700; color: #52738a; }
.hc-empty .hc-empty-text  { font-size: 14px; margin-top: 6px; }
.hc-muted { color: #7590a6; font-size: 13.5px; margin: 0; }
.hc-field-label {
  font-size: 11.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .7px; color: #6b8299; margin-bottom: 5px;
}
.hc-divider { height: 1px; border-top: 1px dashed #e1ecf4; margin: 14px 0; }

/* ---------- tables / plots ---------- */
.hc-scroll { max-height: 420px; overflow: auto; border: 1px solid #e1ecf4;
             border-radius: 9px; }
.hc-scroll table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.hc-scroll th, .hc-scroll td {
  padding: 6px 10px; border-bottom: 1px solid #eef4f8; text-align: right;
  white-space: nowrap;
}
.hc-scroll th { background: #f2f8fc; color: #0b5c85; position: sticky; top: 0;
                font-weight: 700; }
.hc-scroll th:first-child, .hc-scroll td:first-child { text-align: left;
  position: sticky; left: 0; background: #fff; font-weight: 600; }
.hc-scroll th:first-child { background: #f2f8fc; }
.shiny-plot-output { background: #fff; border-radius: 10px; }

/* ---------- v2: profile sub-card inside the sidebar ---------- */
.hc-profile-box {
  background: #f7fbfd; border: 1px solid #dbe9f0; border-radius: 10px;
  padding: 12px 13px; margin-bottom: 12px;
}
.hc-profile-box .hc-field-label { margin-bottom: 4px; }

/* ---------- v2: BMI pill ---------- */
.hc-bmi {
  display: flex; align-items: center; justify-content: space-between;
  background: #eef7fc; border: 1px solid #c9e3f1; border-radius: 9px;
  padding: 9px 12px; margin: 4px 0 12px 0;
}
.hc-bmi .hc-bmi-value { font-size: 19px; font-weight: 700; color: #0b5c85; }
.hc-bmi .hc-bmi-label { font-size: 11.5px; font-weight: 700; letter-spacing: .6px;
                        text-transform: uppercase; color: #6b8299; }
.hc-bmi .hc-bmi-band { font-size: 12.5px; font-weight: 700; }

/* ---------- v2: symptom search + selected chips ---------- */
.hc-search-row { position: relative; margin-bottom: 8px; }
.hc-search-row input { padding-left: 32px; }
.hc-search-icon {
  position: absolute; left: 11px; top: 10px; color: #8fadc2;
  font-size: 14px; pointer-events: none;
}
.hc-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 9px; }
.hc-chip {
  display: inline-block; background: #eaf3f8; color: #0b5c85;
  border: 1px solid #cfe4f0; border-radius: 14px; padding: 2px 9px;
  font-size: 11.5px; font-weight: 600; white-space: nowrap;
}
.hc-chip-hidden { background: #fdf3e2; color: #a15c05; border-color: #f0dcb4; }
.hc-empty-note {
  font-size: 12.5px; color: #7590a6; font-style: italic;
  padding: 6px 0 0 0;
}

/* ---------- v2: Top-3 disease cards ---------- */
.hc-top3 { display: flex; flex-wrap: wrap; gap: 12px; }
.hc-top3-card {
  flex: 1 1 190px; border: 1px solid #e1ecf4; border-radius: 10px;
  padding: 13px 14px; background: #fbfdff;
}
.hc-top3-card.hc-top3-first {
  background: #f2fbf7; border-color: #bfe6d5;
}
.hc-top3-rank {
  font-size: 11px; font-weight: 700; letter-spacing: .7px;
  text-transform: uppercase; color: #6b8299; margin-bottom: 3px;
}
.hc-top3-name {
  font-size: 15.5px; font-weight: 700; color: #0b5c85;
  line-height: 1.25; margin-bottom: 8px; min-height: 38px;
}
.hc-top3-row {
  display: flex; justify-content: space-between; font-size: 12.5px;
  color: #5b7186; padding: 2px 0;
}
.hc-top3-row b { color: #22303f; }
.hc-mini-bar { background: #dbe9f0; border-radius: 10px; height: 7px;
               overflow: hidden; margin: 6px 0 8px 0; }
.hc-mini-bar > div { height: 100%; border-radius: 10px; }

/* ---------- v2: match table ---------- */
.hc-match-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 0; border-bottom: 1px dashed #dbe9f0; font-size: 13.5px;
}
.hc-match-row:last-child { border-bottom: none; }
.hc-match-sym { color: #34495e; }
.hc-tag {
  display: inline-block; border-radius: 12px; padding: 2px 9px;
  font-size: 11px; font-weight: 700; margin-left: 8px;
}
.hc-tag-typical  { background: #e7f7ef; color: #15764f; }
.hc-tag-partial  { background: #fdf3e2; color: #a15c05; }
.hc-tag-unmatched{ background: #f2f6f9; color: #7590a6; }
.hc-score-big {
  font-size: 30px; font-weight: 700; line-height: 1.1; color: #0b5c85;
}

/* ---------- v2: history ---------- */
.hc-scroll table tbody tr:hover { background: #f7fbfd; }
.hc-hist-empty { text-align: center; padding: 26px 18px; color: #7590a6; }
.hc-hist-empty .hc-hist-icon { font-size: 30px; color: #a9c4d6; margin-bottom: 8px; }
.hc-btn-green {
  background: #1e9e6a !important; color: #fff !important; border: none !important;
  font-weight: 700 !important; border-radius: 8px !important;
}
.hc-btn-ghost {
  background: #eef4f8 !important; color: #0b5c85 !important;
  border: 1px solid #cfe0ec !important; font-weight: 600 !important;
  border-radius: 8px !important;
}
.hc-dl-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.hc-dl-row .btn { font-size: 13px; }
@media (max-width: 991px) {
  .hc-hero h1 { font-size: 22px; }
  .hc-symptom-box { max-height: 260px; }
}
"


#' One stat card.
stat_card <- function(value, label, colour_class = "") {
  div(class = paste("hc-stat", colour_class),
      div(class = "hc-stat-value", value),
      div(class = "hc-stat-label", label))
}

#' Small "label above value" block used inside the result cards.
field_block <- function(label, value_tag) {
  div(style = "margin-bottom: 12px;",
      div(class = "hc-field-label", label),
      value_tag)
}

urgency_badge <- function(txt) {
  low  <- tolower(txt)
  cls  <- if (grepl("emergency", low)) "hc-badge hc-badge-red"
          else if (grepl("urgent", low)) "hc-badge hc-badge-red"
          else if (grepl("soon", low))   "hc-badge hc-badge-amber"
          else "hc-badge hc-badge-green"
  tags$span(class = cls, htmltools::htmlEscape(txt))
}

EMPTY_STATE <- div(
  class = "hc-empty",
  div(class = "hc-empty-icon", "\u2695"),
  div(class = "hc-empty-title", "No prediction yet"),
  div(class = "hc-empty-text",
      "Fill in the patient details on the left, tick the symptoms, ",
      "then click ", tags$b("Predict Disease"), ".")
)

DISCLAIMER_HTML <- div(
  class = "hc-note",
  tags$strong("Medical disclaimer: "),
  "This application is a college academic project. The disease prediction and ",
  "the medicine information come from a stored dataset and a statistical model. ",
  "They are shown only for ", tags$b("educational and demonstration purposes"),
  " and are ", tags$b("NOT a medical diagnosis, prescription or professional advice."),
  " Always consult a qualified doctor for real health problems. In an emergency, ",
  "go to the nearest hospital immediately."
)

# ---------------------------------------------------------------------------
# 8.1 Dashboard tab
# ---------------------------------------------------------------------------
dashboard_tab <- tabPanel(
  "Dashboard", icon = icon("home"),
  div(class = "hc-hero",
      h1("Healthcare Recommendation System"),
      p("A machine-learning based web application built in R with Shiny. ",
        "Enter a patient's age, gender and symptoms \u2014 the system predicts the ",
        "most likely disease and then shows the recommended specialist, general ",
        "health information, dataset-based medicine information and precautions.")
  ),

  fluidRow(
    column(3, stat_card(length(DISEASE_LEVELS), "Diseases Covered")),
    column(3, stat_card(format(nrow(HEALTHCARE_DATA), big.mark = ","),
                        "Patient Records", "hc-teal")),
    column(3, stat_card(length(SYMPTOM_COLS), "Symptoms Tracked")),
    column(3, stat_card(
      if (isTRUE(MODEL_INFO$trained_ok)) sprintf("%.1f%%", 100 * MODEL_INFO$test_acc)
      else "N/A",
      "Model Accuracy (test set)", "hc-green"))
  ),

  fluidRow(
    column(8,
      div(class = "hc-card",
        h3("How the system works"),
        div(class = "hc-flow",
          div(class = "hc-step", div(class = "hc-step-no", "1"),
              div(class = "hc-step-title", "Patient Input"),
              div(class = "hc-step-text",
                  "Age, gender and the symptoms the patient has.")),
          div(class = "hc-step", div(class = "hc-step-no", "2"),
              div(class = "hc-step-title", "Preprocessing"),
              div(class = "hc-step-text",
                  "Symptoms are coded 0/1 and aligned to the training columns.")),
          div(class = "hc-step", div(class = "hc-step-no", "3"),
              div(class = "hc-step-title", "ML Prediction"),
              div(class = "hc-step-text",
                  paste("A", if (HAS_RF) "Random Forest" else "decision tree",
                        "classifier returns a disease and a probability."))),
          div(class = "hc-step", div(class = "hc-step-no", "4"),
              div(class = "hc-step-title", "Recommendation"),
              div(class = "hc-step-text",
                  "Specialist, care advice and dataset-based medicine information."))
        )
      ),

      div(class = "hc-card",
        h3("Model information"),
        field_block("Algorithm used",
          tags$span(class = "hc-badge",
                    if (isTRUE(MODEL_INFO$trained_ok)) MODEL_INFO$engine else "not trained")),
        field_block("Training records", tags$b(
          ifelse(is.na(MODEL_INFO$n_train), "-", MODEL_INFO$n_train))),
        field_block("Testing records", tags$b(
          ifelse(is.na(MODEL_INFO$n_test), "-", MODEL_INFO$n_test))),
        field_block("Training accuracy", tags$b(
          ifelse(is.na(MODEL_INFO$train_acc), "-",
                 sprintf("%.2f%%", 100 * MODEL_INFO$train_acc)))),
        field_block("Testing accuracy", tags$b(
          ifelse(is.na(MODEL_INFO$test_acc), "-",
                 sprintf("%.2f%%", 100 * MODEL_INFO$test_acc)))),
        field_block("Cross-validation",
          tags$span(class = "hc-muted",
                    if (HAS_CARET) "caret is installed (confusion matrix statistics available)"
                    else "install package 'caret' for extra statistics")),
        if (!isTRUE(MODEL_INFO$trained_ok))
          div(class = "hc-note",
              tags$strong("Model could not be trained: "), MODEL_INFO$message)
      )
    ),

    column(4,
      div(class = "hc-card",
        h3("Diseases in the dataset"),
        div(class = "hc-scroll", style = "max-height: 300px;",
            tableOutput("dashboard_disease_table"))
      ),
      div(class = "hc-card",
        h3("Quick start"),
        tags$ol(class = "hc-list",
          tags$li("Open the ", tags$b("Predict & Recommend"), " tab."),
          tags$li("Fill the ", tags$b("Patient Profile"), " (age, gender, height, weight)."),
          tags$li("Search and tick the symptoms the patient has."),
          tags$li("Click ", tags$b("Predict Disease"), "."),
          tags$li("Read the Top 3 diseases, the symptom-match score, the specialist ",
                  "and the disease information card."),
          tags$li("See every past prediction under ", tags$b("History"), ", ",
                  "and download it as CSV or an HTML report."),
          tags$li("Charts are in the ", tags$b("Analytics"), " tab.")
        )
      ),
      div(class = "hc-card",
        h3("What's included"),
        tags$ul(class = "hc-list",
          tags$li(tags$b("Patient profile"), " \u2014 age, gender, height, weight, live BMI, smoking, alcohol, existing conditions."),
          tags$li(tags$b("Symptom search"), " \u2014 filter 28 symptoms instantly."),
          tags$li(tags$b("Top 3 predictions"), " with model probability and a symptom-match score."),
          tags$li(tags$b("Disease information card"), " \u2014 common symptoms, causes, precautions."),
          tags$li(tags$b("Session history"), " with CSV and printable report export."),
          tags$li(tags$b("Validation"), " on every input, with all problems listed at once.")
        )
      )
    )
  ),

  DISCLAIMER_HTML
)

# ---------------------------------------------------------------------------
# 8.2 Predict & Recommend tab
# ---------------------------------------------------------------------------
predict_tab <- tabPanel(
  "Predict & Recommend", icon = icon("stethoscope"),

  sidebarLayout(

    sidebarPanel(
      width = 4,

      # ---------------- 1. PATIENT PROFILE -------------------------------
      h4(style = "color:#0b5c85; font-weight:700; margin-top:0;",
         tags$i(class = "fa fa-id-card"), " Patient Profile"),

      fluidRow(
        column(6, numericInput("age", "Age (years)", value = 30,
                               min = 1, max = 120, step = 1)),
        column(6, radioButtons("gender", "Gender",
                               choices = c("Male", "Female"), selected = "Male",
                               inline = TRUE))
      ),

      fluidRow(
        column(6, numericInput("height", "Height (cm)", value = 165,
                               min = 30, max = 250, step = 1)),
        column(6, numericInput("weight", "Weight (kg)", value = 62,
                               min = 5, max = 300, step = 1))
      ),

      uiOutput("bmi_display"),

      fluidRow(
        column(6, radioButtons("smoking", "Smoking",
                               choices = SMOKING_CHOICES, selected = "No")),
        column(6, radioButtons("alcohol", "Alcohol",
                               choices = ALCOHOL_CHOICES, selected = "No"))
      ),

      checkboxGroupInput("conditions", "Existing conditions (if any)",
                         choices = CONDITION_CHOICES, selected = NULL,
                         inline = FALSE),

      numericInput("duration", "How long have the symptoms lasted? (days)",
                   value = 3, min = 0, max = 365, step = 1),

      tags$hr(style = "border-top:1px dashed #e1ecf4; margin:14px 0;"),

      # ---------------- 2. SYMPTOMS WITH SEARCH --------------------------
      tags$div(style = "display:flex; justify-content:space-between; align-items:center;",
        tags$label(tags$i(class = "fa fa-list-check"), " Symptoms",
                   class = "control-label", style = "margin:0;"),
        tags$span(class = "hc-muted", style = "font-size:12px;",
                  textOutput("sym_count", inline = TRUE))
      ),

      tags$div(style = "margin-top:8px;",
        div(class = "hc-search-row",
            tags$i(class = "fa fa-magnifying-glass hc-search-icon"),
            textInput("symptom_search", NULL, placeholder = "Search symptoms...",
                      width = "100%")
        ),
        div(class = "hc-symptom-box",
            checkboxGroupInput(
              "symptoms", label = NULL,
              choices  = stats::setNames(SYMPTOM_COLS, unname(SYMPTOM_LABELS)),
              selected = NULL, width = "100%"
            )
        )
      ),
      uiOutput("selected_chips"),

      div(class = "hc-muted", style = "margin-top:8px;",
          "Tip: select at least 2\u20133 symptoms for a more reliable prediction."),

      tags$hr(style = "border-top:1px dashed #e1ecf4; margin:14px 0;"),

      # ---------------- 3. ACTIONS ---------------------------------------
      actionButton("predict_btn", "\u25B6  Predict Disease",
                   style = paste0("width:100%; background:", COL_PRIMARY,
                                  "; color:#fff; border:none; font-weight:700;",
                                  " padding:11px; font-size:15px; border-radius:8px;")),
      tags$div(style = "height:8px;"),
      actionButton("new_patient_btn", "New Patient",
                   class = "hc-btn-green", style = "width:100%; padding:9px;"),
      tags$div(style = "height:8px;"),
      actionButton("reset_btn", "Reset Form",
                   class = "hc-btn-ghost", style = "width:100%; padding:9px;"),
      tags$div(style = "height:8px;"),
      actionButton("sample_btn", "Load Sample Patient",
                   class = "hc-btn-ghost",
                   style = "width:100%; padding:7px; font-size:13px;")
    ),

    mainPanel(
      width = 8,
      uiOutput("result_ui"),
      DISCLAIMER_HTML
    )
  )
)

# ---------------------------------------------------------------------------
# 8.3 Analytics tab
# ---------------------------------------------------------------------------
analytics_tab <- tabPanel(
  "Analytics", icon = icon("chart-bar"),

  div(class = "hc-hero", style = "padding:20px 24px;",
      h1(style = "font-size:22px;", "Dataset & Model Analytics"),
      p("Three visualisations built with ggplot2: the distribution of diseases in ",
        "the dataset, the most common symptoms, and the per-disease accuracy of the ",
        "trained model on the test data.")
  ),

  fluidRow(
    column(3, stat_card(format(nrow(HEALTHCARE_DATA), big.mark = ","),
                        "Total Records", "hc-teal")),
    column(3, stat_card(
      if (isTRUE(MODEL_INFO$trained_ok)) sprintf("%.2f%%", 100 * MODEL_INFO$test_acc)
      else "N/A", "Overall Test Accuracy", "hc-green")),
    column(3, stat_card(
      ifelse(is.na(MODEL_INFO$n_test), "-", MODEL_INFO$n_test), "Test Records")),
    column(3, stat_card(
      if (!is.null(MODEL_INFO$cm_stats))
        sprintf("%.3f", as.numeric(MODEL_INFO$cm_stats$overall["Kappa"]))
      else "N/A",
      if (is.null(MODEL_INFO$cm_stats)) "Kappa (needs caret)" else "Kappa Statistic"))
  ),

  div(class = "hc-card",
    h3("1. Disease distribution in the dataset"),
    plotOutput("plot_disease_dist", height = "470px")
  ),

  div(class = "hc-card",
    h3("2. Most common symptoms (28 symptoms tracked)"),
    plotOutput("plot_symptom_freq", height = "520px")
  ),

  div(class = "hc-card",
    h3("3. Model performance \u2014 prediction accuracy per disease"),
    plotOutput("plot_model_perf", height = "520px")
  ),

  div(class = "hc-card",
    h3("Confusion matrix (test set)"),
    p(class = "hc-muted",
      "Rows are what the model predicted, columns are the true disease. ",
      "A perfect model would only have numbers on the diagonal."),
    div(class = "hc-scroll", tableOutput("confusion_table"))
  ),

  if (!is.null(MODEL_INFO$importance))
    div(class = "hc-card",
      h3("Bonus \u2014 top 15 most important features for the model"),
      p(class = "hc-muted",
        "Random Forest variable importance (MeanDecreaseAccuracy). ",
        "The list includes age and gender, which are model features too."),
      div(class = "hc-scroll", tableOutput("importance_table"))
    ),

  div(class = "hc-info",
      tags$strong("Reading the charts: "),
      "the accuracy chart only uses the 25% test records that the model has never ",
      "seen during training, so it is an honest estimate of performance.")
)

# ---------------------------------------------------------------------------
# 8.4 Help & About tab
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 8.3b History tab  (v2 features 10 and 11)
# ---------------------------------------------------------------------------
history_tab <- tabPanel(
  "History", icon = icon("clock-rotate-left"),

  div(class = "hc-hero", style = "padding:20px 24px;",
      h1(style = "font-size:22px;", "Prediction History (this session)"),
      p("Every disease prediction you make is stored here for the current browser ",
        "session. Nothing is saved to disk or to a database \u2014 closing the app ",
        "clears it. Download the history as a CSV file below.")
  ),

  fluidRow(
    column(3, stat_card(textOutput("hist_total", inline = TRUE),
                        "Predictions Made", "hc-teal")),
    column(3, stat_card(textOutput("hist_top_disease", inline = TRUE),
                        "Most Frequent Prediction")),
    column(3, stat_card(textOutput("hist_avg_conf", inline = TRUE),
                        "Average Confidence", "hc-green")),
    column(3, stat_card(textOutput("hist_high_conf", inline = TRUE),
                        "High-Confidence Results"))
  ),

  div(class = "hc-card",
    h3(tags$i(class = "fa fa-clock-rotate-left"), " Session records"),
    uiOutput("history_body"),
    div(class = "hc-dl-row",
      downloadButton("download_history", "Download History (CSV)",
                     class = "hc-btn-ghost"),
      downloadButton("download_prediction", "Download Last Prediction (CSV)",
                     class = "hc-btn-ghost"),
      downloadButton("download_report", "Download Report (HTML)",
                     class = "hc-btn-ghost"),
      actionButton("clear_history_btn", "Clear History",
                   class = "hc-btn-ghost")
    ),
    p(class = "hc-muted", style = "margin-top:12px;",
      tags$strong("About the report download: "),
      "a true PDF needs a LaTeX or Pandoc installation alongside R, which is not ",
      "reliable on every machine. Instead the report is downloaded as a single ",
      "self-contained HTML file \u2014 open it in any browser and use ",
      tags$em("Print \u2192 Save as PDF"), " to produce a PDF for your submission.")
  ),

  # --- a disease information browser, reusing the same CSV --------------
  div(class = "hc-card",
    h3(tags$i(class = "fa fa-book-medical"), " Disease information browser"),
    p(class = "hc-muted",
      "Browse everything the dataset knows about any disease, without needing to ",
      "run a prediction. Content is educational only."),
    fluidRow(
      column(5, selectInput("browse_disease", "Choose a disease",
                            choices = DISEASE_LEVELS, selected = DISEASE_LEVELS[1])),
      column(7)
    ),
    uiOutput("browse_disease_ui")
  ),

  DISCLAIMER_HTML
)

help_tab <- tabPanel(
  "Help & About", icon = icon("info-circle"),

  div(class = "hc-card",
    h3("How to use the application"),
    tags$ol(class = "hc-list",
      tags$li("Go to the ", tags$b("Predict & Recommend"), " tab."),
      tags$li("Type the patient's ", tags$b("age"), " and choose the ", tags$b("gender"), "."),
      tags$li("Tick every ", tags$b("symptom"), " the patient has."),
      tags$li("Click ", tags$b("Predict Disease"), "."),
      tags$li("Read the predicted disease, the confidence value, the recommended ",
              "specialist, the general information, the dataset-based medicine ",
              "information and the precautions."),
      tags$li("Use ", tags$b("Reset Form"), " to start over with a new patient."),
      tags$li("Open the ", tags$b("Analytics"), " tab to see the charts for your report.")
    )
  ),

  div(class = "hc-card",
    h3("Machine-learning model"),
    p(class = "hc-muted", "The prediction step is a supervised classification problem."),
    tags$ul(class = "hc-list",
      tags$li(tags$b("Features (29): "), "28 binary symptom columns plus age and gender."),
      tags$li(tags$b("Target: "), "the ", tags$code("Disease"), " column (", length(DISEASE_LEVELS), " classes)."),
      tags$li(tags$b("Algorithm: "),
              if (HAS_RF) "Random Forest with 300 trees."
              else "Decision tree (rpart). Install randomForest for better accuracy."),
      tags$li(tags$b("Why Random Forest? "),
              "It handles many binary inputs, needs no scaling, resists overfitting ",
              "through bagging, and gives probabilities plus variable importance."),
      tags$li(tags$b("Validation: "), "stratified 75/25 train-test split with a fixed seed."),
      tags$li(tags$b("Output: "), "the class with the highest probability, together with ",
              "that probability \u2014 shown as the confidence percentage.")
    )
  ),

  div(class = "hc-card",
    h3("Recommendation system"),
    p(class = "hc-muted",
      "The recommendation layer is a simple rule-based lookup, not a second model."),
    tags$ol(class = "hc-list",
      tags$li("The model returns a predicted disease label."),
      tags$li("That label is used to look up the matching row of ", tags$code("recommendations.csv"), "."),
      tags$li("The row supplies the specialist, the disease information, the general ",
              "medicine information, home care, precautions and an urgency level."),
      tags$li("If a disease has no row in the file, a safe generic recommendation ",
              "(\u201cGeneral Physician\u201d) is shown instead, so the app never crashes.")
    )
  ),

  div(class = "hc-card",
    h3("Install packages & run"),
    p(class = "hc-muted", "Run once in the R console:"),
    tags$pre(style = "background:#f2f8fc; border:1px solid #e1ecf4; border-radius:8px; padding:12px; font-size:13px;",
             'install.packages(c("shiny", "dplyr", "ggplot2", "randomForest", "caret"))'),
    p(class = "hc-muted", "Then start the app:"),
    tags$pre(style = "background:#f2f8fc; border:1px solid #e1ecf4; border-radius:8px; padding:12px; font-size:13px;",
             'setwd("path/to/Healthcare-Recommendation")\nshiny::runApp()')
  ),

  DISCLAIMER_HTML
)

# ---------------------------------------------------------------------------
# 8.5 Assemble the page
# ---------------------------------------------------------------------------
ui <- navbarPage(
  title       = tags$span("\u2695 Healthcare Recommendation System"),
  windowTitle = "Healthcare Recommendation System",
  id          = "main_nav",
  collapsible = TRUE,
  header      = tags$head(tags$style(htmltools::HTML(APP_CSS))),
  dashboard_tab,
  predict_tab,
  analytics_tab,
  history_tab,
  help_tab,
  footer = tags$div(
    style = "text-align:center; padding:18px; color:#7590a6; font-size:12.5px;",
    "Healthcare Recommendation System using R \u2014 academic project. ",
    "Not a substitute for professional medical advice."
  )
)


# ============================================================================
# SECTION 9 - SERVER
# ============================================================================

server <- function(input, output, session) {

  # -------------------------------------------------------------------------
  # 9.1 Dashboard: disease list table
  # -------------------------------------------------------------------------
  output$dashboard_disease_table <- renderTable({
    df <- data.frame(
      Disease = DISEASE_DIST$Disease,
      Records = DISEASE_DIST$Records,
      stringsAsFactors = FALSE
    )
    names(df) <- c("Disease", "Records")
    df
  }, striped = FALSE, hover = TRUE, bordered = FALSE, spacing = "xs",
     align = "lr", width = "100%")

  # -------------------------------------------------------------------------
  # 9.1b Live BMI read-out (v2 feature 1)
  # -------------------------------------------------------------------------
  output$bmi_display <- renderUI({
    bmi <- bmi_value(input$height, input$weight)
    band <- bmi_band(bmi)
    div(class = "hc-bmi",
      div(
        div(class = "hc-bmi-label", "Body Mass Index"),
        div(class = "hc-bmi-value",
            if (is.na(bmi)) "--" else sprintf("%.1f", bmi))
      ),
      div(class = "hc-bmi-band",
          style = paste0("color:", band$colour, ";"), band$label)
    )
  })

  # -------------------------------------------------------------------------
  # 9.2 Symptom selection with search  (v2 feature 2)
  # -------------------------------------------------------------------------
  # `selected_syms` is the MASTER list of ticked symptoms. It deliberately keeps
  # symptoms that the current search text is hiding, so a search never silently
  # drops an earlier selection.
  selected_syms <- reactiveVal(character(0))

  # The symptom choices currently visible under the search box.
  visible_choices <- reactive({
    q <- trimws(tolower(input$symptom_search %||% ""))
    all_ch <- stats::setNames(SYMPTOM_COLS, unname(SYMPTOM_LABELS))
    if (!nzchar(q)) return(all_ch)
    keep <- grepl(q, tolower(unname(all_ch)), fixed = TRUE) |
            grepl(q, tolower(names(all_ch)),  fixed = TRUE)
    all_ch[keep]
  })

  # The symptoms ticked by the user, RIGHT NOW.
  #   = what is ticked on screen  +  anything ticked earlier that the current
  #     search text is hiding.
  # This is read synchronously (never via a stored observer result), so a
  # prediction can never depend on whether another observer happened to run
  # first.
  current_symptoms <- reactive({
    vis    <- unname(visible_choices())
    hidden <- setdiff(selected_syms(), vis)     # ticked earlier, now filtered out
    unique(c(hidden, as.character(input$symptoms %||% character(0))))
  })

  # Re-render the checkbox list whenever the search text changes.
  # NOTE: visible_choices() is a NAMED vector - the names are the display labels
  # ("Joint Pain") and the values are the internal keys ("joint_pain"). A
  # checkboxGroupInput reports the VALUES, so the already-ticked set must be
  # matched against unname(ch), never names(ch).
  observeEvent(visible_choices(), {
    ch  <- visible_choices()
    sel <- intersect(selected_syms(), unname(ch))
    updateCheckboxGroupInput(session, "symptoms",
                             choices = ch, selected = sel)
  }, ignoreNULL = FALSE, ignoreInit = TRUE)

  # Remember the ticked set (including items the search is currently hiding) so
  # that clearing the search box restores them.
  observeEvent(list(input$symptoms, visible_choices()), {
    vis    <- unname(visible_choices())
    hidden <- setdiff(selected_syms(), vis)
    selected_syms(unique(c(hidden, as.character(input$symptoms %||% character(0)))))
  }, ignoreNULL = FALSE, ignoreInit = TRUE)

  output$sym_count <- renderText({
    n <- length(current_symptoms())
    paste0(n, if (n == 1L) " symptom selected" else " symptoms selected")
  })

  # Show the selection as chips so hidden ones stay visible.
  output$selected_chips <- renderUI({
    sel <- current_symptoms()
    if (length(sel) == 0L) {
      return(div(class = "hc-empty-note", "No symptoms selected yet."))
    }
    vis    <- unname(visible_choices())
    hidden <- setdiff(sel, vis)
    lab    <- unname(SYMPTOM_LABELS[sel])
    lab[is.na(lab)] <- sel[is.na(lab)]

    div(class = "hc-chips",
      lapply(seq_along(lab), function(i) {
        is_hidden <- sel[i] %in% hidden
        tags$span(
          class = paste("hc-chip", if (is_hidden) "hc-chip-hidden" else ""),
          title = if (is_hidden) "Selected but hidden by the current search" else NULL,
          lab[i]
        )
      })
    )
  })

  # -------------------------------------------------------------------------
  # 9.3 Prediction result + session history
  # -------------------------------------------------------------------------
  pred_result <- reactiveVal(NULL)

  # Empty history table with fixed column types, so rbind always works.
  empty_history <- function() {
    data.frame(
      Patient    = integer(0),
      Time       = character(0),
      Age        = integer(0),
      Gender     = character(0),
      BMI        = character(0),
      Symptoms   = integer(0),
      Predicted  = character(0),
      Confidence = character(0),
      Second     = character(0),
      Prob_2     = character(0),
      Third      = character(0),
      Prob_3     = character(0),
      Match      = character(0),
      Specialist = character(0),
      Urgency    = character(0),
      stringsAsFactors = FALSE
    )
  }
  history_df  <- reactiveVal(empty_history())
  patient_no  <- reactiveVal(0L)

  # -------------------------------------------------------------------------
  # 9.3a Turn raw inputs into a clean patient record, collecting every problem
  #      instead of stopping at the first one (v2 feature 12)
  # -------------------------------------------------------------------------
  read_patient_inputs <- function() {
    errs <- character(0)

    num_in <- function(value, label, lo, hi, allow_na = FALSE) {
      v <- suppressWarnings(as.numeric(value))
      if (length(v) == 0L || !is.finite(v)) {
        if (!allow_na) errs <<- c(errs, sprintf("%s is missing or not a number.", label))
        return(NA_real_)
      }
      if (v < lo || v > hi) {
        errs <<- c(errs, sprintf("%s must be between %g and %g (you entered %g).",
                                 label, lo, hi, v))
        return(NA_real_)
      }
      v
    }

    age      <- num_in(input$age,      "Age",        1,  120)
    height   <- num_in(input$height,   "Height (cm)", 30, 250)
    weight   <- num_in(input$weight,   "Weight (kg)", 5,  300)
    duration <- num_in(input$duration, "Symptom duration", 0, 365)

    gender <- input$gender %||% "Male"
    if (!gender %in% c("Male", "Female")) {
      errs <- c(errs, "Gender must be Male or Female.")
      gender <- "Male"
    }

    syms <- intersect(current_symptoms(), SYMPTOM_COLS)
    if (length(syms) == 0L) {
      errs <- c(errs, paste0("No symptoms selected. Please tick at least one ",
                             "symptom (2\u20133 is better) before predicting."))
    }

    list(ok = length(errs) == 0L, errors = errs, age = age, gender = gender,
         height = height, weight = weight, duration = duration,
         symptoms = syms, smoking = input$smoking %||% "No",
         alcohol = input$alcohol %||% "No",
         conditions = input$conditions %||% character(0))
  }

  # -------------------------------------------------------------------------
  # 9.3b Run a prediction and store the result
  # -------------------------------------------------------------------------
  do_predict <- function() {

    inp <- read_patient_inputs()
    if (!inp$ok) {
      pred_result(list(ok = FALSE, msgs = inp$errors))
      return(invisible(FALSE))
    }

    # ---- build the one-row patient record ----------------------------------
    patient <- as.data.frame(
      matrix(0L, nrow = 1L, ncol = length(SYMPTOM_COLS),
             dimnames = list(NULL, SYMPTOM_COLS)), stringsAsFactors = FALSE)
    patient[1, inp$symptoms] <- 1L
    patient$Age    <- round(inp$age)
    patient$Gender <- inp$gender

    result <- tryCatch({
      features <- prepare_features(patient, default_age = 30)
      p        <- predict_patient(MODEL_INFO, features)
      rec      <- get_recommendation(p$disease)
      match    <- match_disease_to_symptoms(p$disease, inp$symptoms)

      list(ok = TRUE, msgs = character(0),
           disease = p$disease, confidence = p$confidence, ranking = p$ranking,
           rec = rec, match = match,
           age = round(inp$age), gender = inp$gender,
           height = inp$height, weight = inp$weight, duration = inp$duration,
           bmi = bmi_value(inp$height, inp$weight),
           smoking = inp$smoking, alcohol = inp$alcohol,
           conditions = inp$conditions,
           n_symptoms = length(inp$symptoms), symptoms = inp$symptoms,
           time = format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
    }, error = function(e) {
      list(ok = FALSE,
           msgs = paste0("The prediction could not be completed: ",
                         conditionMessage(e),
                         " Please check that all packages are installed and that ",
                         "healthcare_data.csv is in the app folder."))
    })

    if (isTRUE(result$ok)) {
      # ---- append to session history ---------------------------------------
      patient_no(patient_no() + 1L)
      result$patient_no <- patient_no()

      rk <- result$ranking
      prob_of <- function(i) if (nrow(rk) >= i) rk$Disease[i] else "-"
      pct_of  <- function(i) if (nrow(rk) >= i)
        sprintf("%.1f%%", 100 * rk$Probability[i]) else "-"

      history_df(rbind(
        history_df(),
        data.frame(
          Patient    = result$patient_no,
          Time       = result$time,
          Age        = result$age,
          Gender     = result$gender,
          BMI        = if (is.na(result$bmi)) "-" else sprintf("%.1f (%.0fcm/%.0fkg)",
                          result$bmi, result$height, result$weight),
          Symptoms   = result$n_symptoms,
          Predicted  = result$disease,
          Confidence = sprintf("%.1f%%", 100 * result$confidence),
          Second     = prob_of(2),
          Prob_2     = pct_of(2),
          Third      = prob_of(3),
          Prob_3     = pct_of(3),
          Match      = if (is.na(result$match$score)) "-" else
                          sprintf("%.1f%%", 100 * result$match$score),
          Specialist = rec_field(result$rec, "Specialist", "General Physician"),
          Urgency    = rec_field(result$rec, "Urgency_Level", "Consult a doctor"),
          stringsAsFactors = FALSE
        )
      ))
    }

    pred_result(result)
    invisible(isTRUE(result$ok))
  }

  observeEvent(input$predict_btn, do_predict())

  # -------------------------------------------------------------------------
  # 9.3c Load an example patient - speeds up a live demonstration
  # -------------------------------------------------------------------------
  observeEvent(input$sample_btn, {
    updateNumericInput(session, "age",      value = 34)
    updateRadioButtons(session, "gender",   selected = "Female")
    updateNumericInput(session, "height",   value = 160)
    updateNumericInput(session, "weight",   value = 58)
    updateRadioButtons(session, "smoking",   selected = "No")
    updateRadioButtons(session, "alcohol",   selected = "No")
    updateCheckboxGroupInput(session, "conditions", selected = "None")
    updateNumericInput(session, "duration", value = 4)
    updateTextInput(session, "symptom_search", value = "")

    demo <- c("fever", "joint_pain", "body_pain", "headache", "skin_rash")
    selected_syms(demo)
    updateCheckboxGroupInput(session, "symptoms",
      choices = stats::setNames(SYMPTOM_COLS, unname(SYMPTOM_LABELS)),
      selected = demo)

    pred_result(NULL)
    showNotification("Sample patient loaded. Click Predict Disease.",
                     type = "message", duration = 4)
  })

  # -------------------------------------------------------------------------
  # 9.3d Reset Form (keeps the patient profile) and New Patient (clears all)
  # -------------------------------------------------------------------------
  clear_result_only <- function() pred_result(NULL)

  observeEvent(input$reset_btn, {
    # Reset clears the SYMPTOMS and the result, but keeps the patient profile so
    # the same patient can quickly be re-tested with a different symptom set.
    selected_syms(character(0))
    updateTextInput(session, "symptom_search", value = "")
    updateCheckboxGroupInput(session, "symptoms",
      choices = stats::setNames(SYMPTOM_COLS, unname(SYMPTOM_LABELS)),
      selected = character(0))
    clear_result_only()
    showNotification("Symptoms cleared. Patient profile kept.",
                     type = "message", duration = 3)
  })

  observeEvent(input$new_patient_btn, {
    # New Patient clears absolutely everything, ready for the next person.
    updateNumericInput(session, "age",      value = 30)
    updateRadioButtons(session, "gender",   selected = "Male")
    updateNumericInput(session, "height",   value = 165)
    updateNumericInput(session, "weight",   value = 62)
    updateRadioButtons(session, "smoking",   selected = "No")
    updateRadioButtons(session, "alcohol",   selected = "No")
    updateCheckboxGroupInput(session, "conditions", selected = character(0))
    updateNumericInput(session, "duration", value = 3)
    updateTextInput(session, "symptom_search", value = "")
    selected_syms(character(0))
    updateCheckboxGroupInput(session, "symptoms",
      choices = stats::setNames(SYMPTOM_COLS, unname(SYMPTOM_LABELS)),
      selected = character(0))
    clear_result_only()
    showNotification(
      sprintf("Ready for a new patient. %d prediction(s) in history.",
              nrow(history_df())), type = "message", duration = 4)
  })

  # -------------------------------------------------------------------------
  # 9.4 Render the result panel
  # -------------------------------------------------------------------------
  output$result_ui <- renderUI({

    r <- pred_result()

    # --- nothing pressed yet -----------------------------------------------
    if (is.null(r)) return(EMPTY_STATE)

    # --- validation / error card (may list several problems at once) -------
    if (!isTRUE(r$ok)) {
      msgs <- r$msgs %||% r$msg %||% "Unknown input problem."
      return(div(class = "hc-note",
        tags$strong(if (length(msgs) == 1L) "Cannot predict yet. "
                    else "Cannot predict yet \u2014 please fix the following:"),
        if (length(msgs) == 1L) htmltools::htmlEscape(msgs)
        else tags$ul(style = "margin:7px 0 0 0;",
                     lapply(msgs, function(m) tags$li(htmltools::htmlEscape(m))))
      ))
    }

    rec <- r$rec
    conf_pct <- 100 * r$confidence
    bar_col  <- confidence_colour(r$confidence)

    symptom_names <- unname(SYMPTOM_LABELS[r$symptoms])
    symptom_names <- symptom_names[!is.na(symptom_names)]

    # ---- optional low-confidence warning -----------------------------------
    low_conf_note <- if (r$confidence < 0.40) {
      div(class = "hc-note",
        tags$strong("Low confidence prediction. "),
        sprintf(paste0("The model is only %.1f%% sure about this result because the ",
                       "selected symptoms are shared by several diseases. "), conf_pct),
        "Try adding one or two more distinctive symptoms (for example a skin rash, ",
        "night sweats or frequent urination) for a sharper prediction. ",
        "Please treat this result as a hint only and consult a doctor.")
    }

    div(
      # ---------------- prediction card ----------------------------------
      div(class = "hc-result",
        div(class = "hc-result-label", "Predicted disease"),
        div(class = "hc-result-name", htmltools::htmlEscape(r$disease)),
        div(class = "hc-bar-wrap",
            div(class = "hc-bar",
                style = paste0("width:", sprintf("%.1f", conf_pct),
                               "%; background:", bar_col, ";"))),
        div(class = "hc-bar-text",
            sprintf("Confidence: %.1f%%", conf_pct)),
        div(class = "hc-divider"),
        fluidRow(
          column(4, field_block("Patient",
                                tags$b(sprintf("%d yrs, %s", r$age, r$gender)))),
          column(4, field_block("Symptoms given",
                                tags$b(sprintf("%d", r$n_symptoms)))),
          column(4, field_block("Prediction certainty",
                                tags$span(class = "hc-badge",
                                  if (r$confidence >= 0.70) "High"
                                  else if (r$confidence >= 0.40) "Medium"
                                  else "Low")))
        ),
        field_block("Symptoms selected",
                    tags$span(class = "hc-muted",
                              htmltools::htmlEscape(paste(symptom_names, collapse = ", "))))
      ),

      low_conf_note,

      # ---------------- TOP 3 with match scores (features 4 and 7) --------
      div(class = "hc-card",
        h3(tags$i(class = "fa fa-ranking-star"), " Top 3 possible diseases"),
        p(class = "hc-muted", style = "margin-bottom:14px;",
          "The model gives a probability for every disease. The three highest are ",
          "shown below, each with a symptom-match score: how typical your selected ",
          "symptoms are of that disease in the dataset."),
        div(class = "hc-top3",
          lapply(seq_len(nrow(r$ranking)), function(i) {
            dis   <- r$ranking$Disease[i]
            prob  <- 100 * r$ranking$Probability[i]
            mt    <- match_disease_to_symptoms(dis, r$symptoms)
            score <- if (is.na(mt$score)) 0 else 100 * mt$score
            s_col <- if (score >= 60) COL_GREEN else if (score >= 35) COL_AMBER else COL_RED

            div(class = paste("hc-top3-card", if (i == 1L) "hc-top3-first" else ""),
              div(class = "hc-top3-rank",
                  sprintf("#%d %s", i, if (i == 1L) "\u2014 most likely" else "")),
              div(class = "hc-top3-name", htmltools::htmlEscape(dis)),
              div(class = "hc-mini-bar",
                  div(style = paste0("width:", sprintf("%.1f", prob),
                                     "%; background:", COL_PRIMARY, ";"))),
              div(class = "hc-top3-row",
                  tags$span("Model probability"), tags$b(sprintf("%.1f%%", prob))),
              div(class = "hc-top3-row",
                  tags$span("Symptom match"),
                  tags$b(style = paste0("color:", s_col, ";"),
                         if (is.na(mt$score)) "-" else sprintf("%.0f%%", score))),
              div(class = "hc-top3-row",
                  tags$span("Typical symptoms"),
                  tags$b(sprintf("%d of %d", mt$matched, r$n_symptoms)))
            )
          })
        )
      ),

      # ---------------- symptom matching detail (feature 7) ---------------
      div(class = "hc-card",
        h3(tags$i(class = "fa fa-crosshairs"), " Symptom matching score"),
        fluidRow(
          column(4,
            div(class = "hc-score-big",
                if (is.na(r$match$score)) "-"
                else sprintf("%.0f%%", 100 * r$match$score)),
            div(class = "hc-field-label", "Match score for"),
            tags$b(htmltools::htmlEscape(r$disease))
          ),
          column(8,
            p(style = "font-size:14px; line-height:1.6; color:#34495e; margin-top:6px;",
              htmltools::htmlEscape(match_sentence(r$match, r$n_symptoms)),
              " The score is the average of P(symptom | disease) taken from the ",
              "dataset for the symptoms you ticked."),
            div(style = "margin-top:6px;",
              tags$span(class = "hc-tag hc-tag-typical",
                        sprintf("%d typical", r$match$matched)),
              tags$span(class = "hc-tag hc-tag-partial",
                        sprintf("%d sometimes seen", r$match$partial)),
              tags$span(class = "hc-tag hc-tag-unmatched",
                        sprintf("%d not typical", r$match$unmatched))
            )
          )
        ),
        div(class = "hc-divider"),
        lapply(seq_len(nrow(r$match$detail)), function(i) {
          lvl <- r$match$detail$Level[i]
          cls <- switch(lvl,
            "Typical"        = "hc-tag hc-tag-typical",
            "Sometimes seen" = "hc-tag hc-tag-partial",
            "hc-tag hc-tag-unmatched")
          div(class = "hc-match-row",
            tags$span(class = "hc-match-sym",
                      htmltools::htmlEscape(r$match$detail$Symptom[i])),
            tags$span(
              tags$span(class = "hc-muted",
                        sprintf("%.0f%% of patients", 100 * r$match$detail$Typicality[i])),
              tags$span(class = cls, lvl)
            )
          )
        }),
        p(class = "hc-muted", style = "margin-top:10px;",
          "A high match score means your symptoms fit this disease well. A low ",
          "score with a high model probability is a warning sign \u2014 please ",
          "consult a doctor.")
      ),

      # ---------------- recommendation cards ------------------------------
      div(class = "hc-card",
        h3(tags$i(class = "fa fa-user-doctor"), " Healthcare Recommendation"),
        fluidRow(
          column(6,
            field_block("Recommended specialist",
              tags$span(class = "hc-badge",
                        htmltools::htmlEscape(rec_field(rec, "Specialist",
                                                        "General Physician"))))
          ),
          column(6,
            field_block("Action advised",
              tags$b(htmltools::htmlEscape(rec_field(rec, "Urgency_Level",
                                                     "Consult a doctor")))))
        ),
        field_block("About this condition",
          tags$p(style = "font-size:14px; line-height:1.6; color:#34495e; margin:0;",
                 htmltools::htmlEscape(rec_field(rec, "Description"))))
      ),

      # ---------------- disease information card (feature 8) --------------
      div(class = "hc-card",
        h3(tags$i(class = "fa fa-book-medical"), " Disease information card"),
        fluidRow(
          column(6,
            field_block("Common symptoms of this disease",
                        bullet_list(rec_field(rec, "Common_Symptoms")))
          ),
          column(6,
            field_block("Common causes",
                        bullet_list(rec_field(rec, "Causes")))
          )
        ),
        div(class = "hc-divider"),
        fluidRow(
          column(6,
            field_block("Patients in this dataset",
              tags$b(sprintf("%s records",
                             format(DISEASE_COUNTS[[r$disease]] %||% 0,
                                    big.mark = ","))))
          ),
          column(6,
            field_block("Symptoms in the dataset",
              tags$b(sprintf("%d of %d tracked",
                             sum(!is.na(SYMPTOM_PROFILE[r$disease, ]) &
                                   SYMPTOM_PROFILE[r$disease, ] >= MATCH_TYPICAL_CUTOFF),
                             length(SYMPTOM_COLS)))))
        )
      ),

      div(class = "hc-card",
        h4("Medicine information (educational \u2014 from the dataset)"),
        div(class = "hc-note", style = "margin-bottom:12px;",
          tags$strong("Not medical advice. "),
          "The list below is general, publicly documented information stored in ",
          "recommendations.csv for academic demonstration. It is ",
          tags$b("not a prescription"), " and does not replace a doctor. ",
          tags$b("Never self-medicate.")),
        bullet_list(rec_field(rec, "Medicines_Educational"))
      ),

      div(class = "hc-card",
        h4(tags$i(class = "fa fa-house-medical"), " Home care"),
        bullet_list(rec_field(rec, "Home_Care"))
      ),

      div(class = "hc-card",
        h4(tags$i(class = "fa fa-triangle-exclamation"),
           " When to see a doctor / precautions"),
        bullet_list(rec_field(rec, "Precautions"))
      ),

      # ---------------- patient profile summary (feature 1) ---------------
      div(class = "hc-card",
        h4(tags$i(class = "fa fa-id-card"), " Patient profile recorded"),
        fluidRow(
          column(3, field_block("Age / Gender",
                                tags$b(sprintf("%d yrs, %s", r$age, r$gender)))),
          column(3, field_block("Height / Weight",
                                tags$b(sprintf("%.0f cm, %.0f kg", r$height, r$weight)))),
          column(3, field_block("BMI",
                                tags$b(sprintf("%s (%s)",
                                  if (is.na(r$bmi)) "-" else sprintf("%.1f", r$bmi),
                                  bmi_band(r$bmi)$label)))),
          column(3, field_block("Symptoms lasted",
                                tags$b(sprintf("%.0f day%s", r$duration,
                                               if (r$duration == 1) "" else "s"))))
        ),
        fluidRow(
          column(4, field_block("Smoking", tags$b(htmltools::htmlEscape(r$smoking)))),
          column(4, field_block("Alcohol", tags$b(htmltools::htmlEscape(r$alcohol)))),
          column(4, field_block("Existing conditions",
                    tags$b(if (length(r$conditions) == 0L) "None reported"
                           else htmltools::htmlEscape(paste(r$conditions, collapse = ", ")))))
        ),
        p(class = "hc-muted", style = "margin-top:6px;",
          "BMI band uses the standard WHO ranges and is shown for general ",
          "awareness only, not as a diagnosis.")
      )
    )
  })

  # -------------------------------------------------------------------------
  # 9.5 Analytics plots
  # -------------------------------------------------------------------------
  output$plot_disease_dist <- renderPlot({
    validate(need(nrow(DISEASE_DIST) > 0, "No data available."))

    ggplot(DISEASE_DIST, aes(x = reorder(Disease, Records), y = Records)) +
      geom_col(aes(fill = Records), width = 0.68, show.legend = FALSE) +
      geom_text(aes(label = Records), hjust = -0.22, size = 3.6, colour = "#4a6377") +
      coord_flip() +
      scale_fill_gradient(low = "#a9dcec", high = "#0b5c85") +
      scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
      labs(title = "Number of patient records per disease",
           subtitle = paste0("Total records: ",
                             format(nrow(HEALTHCARE_DATA), big.mark = ","),
                             "  |  Diseases: ", nrow(DISEASE_DIST)),
           x = NULL, y = "Number of patient records") +
      theme_minimal(base_size = 13) +
      theme(panel.grid.major.y = element_blank(),
            panel.grid.minor = element_blank(),
            plot.title = element_text(face = "bold", colour = "#0b5c85", size = 15),
            plot.subtitle = element_text(colour = "#6b8299", size = 11.5),
            axis.title.x = element_text(colour = "#4a6377"),
            axis.text = element_text(colour = "#54718a"))
  }, res = 100)

  output$plot_symptom_freq <- renderPlot({
    validate(need(nrow(SYMPTOM_DIST) > 0, "No data available."))

    top_n <- min(15L, nrow(SYMPTOM_DIST))
    d <- SYMPTOM_DIST[seq_len(top_n), , drop = FALSE]
    d$Symptom <- factor(d$Symptom, levels = rev(d$Symptom))

    ggplot(d, aes(x = Symptom, y = Percent)) +
      geom_col(fill = "#0b7fa8", width = 0.68) +
      geom_text(aes(label = sprintf("%.1f%%", Percent)),
                hjust = -0.18, size = 3.5, colour = "#4a6377") +
      coord_flip() +
      scale_y_continuous(expand = expansion(mult = c(0, 0.14)),
                         limits = c(0, max(d$Percent) * 1.12)) +
      labs(title = "Most common symptoms in the dataset",
           subtitle = paste0("Top ", top_n, " of ", nrow(SYMPTOM_DIST),
                             " tracked symptoms  |  share of all ",
                             format(nrow(HEALTHCARE_DATA), big.mark = ","),
                             " patient records"),
           x = NULL, y = "Patients showing the symptom (%)") +
      theme_minimal(base_size = 13) +
      theme(panel.grid.major.y = element_blank(),
            panel.grid.minor = element_blank(),
            plot.title = element_text(face = "bold", colour = "#0b5c85", size = 15),
            plot.subtitle = element_text(colour = "#6b8299", size = 11.5),
            axis.title.x = element_text(colour = "#4a6377"),
            axis.text = element_text(colour = "#54718a"))
  }, res = 100)

  output$plot_model_perf <- renderPlot({
    pc <- MODEL_INFO$per_class
    validate(need(isTRUE(MODEL_INFO$trained_ok) && nrow(pc) > 0,
                  "The model is not available, so performance charts cannot be drawn."))

    pc$Pct <- 100 * pc$Accuracy
    pc$Disease <- factor(pc$Disease, levels = pc$Disease)   # already sorted

    ggplot(pc, aes(x = Disease, y = Pct)) +
      geom_col(aes(fill = Pct), width = 0.68, show.legend = FALSE) +
      geom_text(aes(label = sprintf("%.0f%%", Pct)),
                hjust = -0.2, size = 3.5, colour = "#4a6377") +
      geom_hline(yintercept = 100 * MODEL_INFO$test_acc,
                 linetype = "dashed", colour = "#d64545", linewidth = 0.7) +
      annotate("text", x = 0.65, y = 100 * MODEL_INFO$test_acc,
               label = sprintf("overall accuracy %.1f%%", 100 * MODEL_INFO$test_acc),
               hjust = 0, vjust = 1.6, size = 3.5, colour = "#d64545") +
      coord_flip() +
      scale_fill_gradient(low = "#f6d3ab", high = "#1e9e6a") +
      scale_y_continuous(limits = c(0, 108), expand = expansion(mult = c(0, 0))) +
      labs(title = "Prediction accuracy for each disease (test set)",
           subtitle = paste0("Algorithm: ", MODEL_INFO$engine,
                             "  |  train records: ", MODEL_INFO$n_train,
                             "  |  test records: ", MODEL_INFO$n_test),
           x = NULL, y = "Accuracy (%)") +
      theme_minimal(base_size = 13) +
      theme(panel.grid.major.y = element_blank(),
            panel.grid.minor = element_blank(),
            plot.title = element_text(face = "bold", colour = "#0b5c85", size = 15),
            plot.subtitle = element_text(colour = "#6b8299", size = 11.5),
            axis.title.x = element_text(colour = "#4a6377"),
            axis.text = element_text(colour = "#54718a"))
  }, res = 100)

  # -------------------------------------------------------------------------
  # 9.6 Confusion matrix + importance tables
  # -------------------------------------------------------------------------
  output$confusion_table <- renderTable({
    validate(need(isTRUE(MODEL_INFO$trained_ok) && !is.null(MODEL_INFO$cm_table),
                  "Confusion matrix is not available."))
    cm <- MODEL_INFO$cm_table
    df <- as.data.frame.matrix(cm)
    if (nrow(df) == 0L) return(NULL)
    df <- cbind(data.frame(Predicted = rownames(df), stringsAsFactors = FALSE), df)
    df
  }, striped = FALSE, hover = TRUE, bordered = FALSE, spacing = "xs",
     align = "l", width = "100%", digits = 0)

  output$importance_table <- renderTable({
    imp <- MODEL_INFO$importance
    if (is.null(imp)) return(NULL)
    d <- head(imp, 15)
    d$Feature <- unname(ifelse(d$Feature %in% names(SYMPTOM_LABELS),
                               SYMPTOM_LABELS[d$Feature], d$Feature))
    d
  }, striped = FALSE, hover = TRUE, bordered = FALSE, spacing = "xs",
     align = "lr", width = "100%")

  # -------------------------------------------------------------------------
  # 9.7 Session history  (v2 feature 10)
  # -------------------------------------------------------------------------
  output$hist_total <- renderText({
    n <- nrow(history_df())
    format(n, big.mark = ",")
  })

  output$hist_top_disease <- renderText({
    h <- history_df()
    if (nrow(h) == 0L) return("-")
    tb <- sort(table(h$Predicted), decreasing = TRUE)
    names(tb)[1]
  })

  output$hist_avg_conf <- renderText({
    h <- history_df()
    if (nrow(h) == 0L) return("-")
    v <- suppressWarnings(as.numeric(sub("%", "", h$Confidence)))
    v <- v[is.finite(v)]
    if (length(v) == 0L) return("-")
    sprintf("%.1f%%", mean(v))
  })

  output$hist_high_conf <- renderText({
    h <- history_df()
    if (nrow(h) == 0L) return("0")
    v <- suppressWarnings(as.numeric(sub("%", "", h$Confidence)))
    sum(is.finite(v) & v >= 70)
  })

  output$history_body <- renderUI({
    h <- history_df()
    if (nrow(h) == 0L) {
      return(div(class = "hc-hist-empty",
        div(class = "hc-hist-icon", "\u23F1"),
        div(tags$b("No predictions yet")),
        div(class = "hc-muted",
            "Go to the Predict & Recommend tab, run a prediction, and it will ",
            "appear here automatically.")
      ))
    }
    div(class = "hc-scroll",
      tableOutput("history_table"))
  })

  output$history_table <- renderTable({
    h <- history_df()
    if (nrow(h) == 0L) return(NULL)
    # newest first
    h[rev(seq_len(nrow(h))), , drop = FALSE]
  }, striped = FALSE, hover = TRUE, bordered = FALSE, spacing = "xs",
     align = "c", width = "100%")

  observeEvent(input$clear_history_btn, {
    history_df(empty_history())
    showNotification("Session history cleared.", type = "warning", duration = 3)
  })

  # -------------------------------------------------------------------------
  # 9.8 Downloads  (v2 feature 11)
  # -------------------------------------------------------------------------
  output$download_history <- downloadHandler(
    filename = function() {
      paste0("healthcare_prediction_history_",
             format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv")
    },
    content = function(file) {
      h <- history_df()
      if (nrow(h) == 0L) {
        h <- data.frame(Message = "No predictions were made in this session.")
      }
      utils::write.csv(h, file, row.names = FALSE)
    }
  )

  output$download_prediction <- downloadHandler(
    filename = function() {
      paste0("healthcare_prediction_",
             format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv")
    },
    content = function(file) {
      r <- pred_result()
      if (is.null(r) || !isTRUE(r$ok)) {
        utils::write.csv(
          data.frame(Message = "No successful prediction is currently displayed."),
          file, row.names = FALSE)
        return()
      }
      df <- data.frame(
        Field = c("Predicted disease", "Confidence", "Second possibility",
                  "Second probability", "Third possibility", "Third probability",
                  "Symptom match score", "Typical symptoms matched",
                  "Age", "Gender", "Height_cm", "Weight_kg", "BMI", "BMI band",
                  "Symptom duration (days)", "Smoking", "Alcohol",
                  "Existing conditions", "Symptoms given (count)",
                  "Symptoms", "Recommended specialist", "Action advised",
                  "Model", "Model test accuracy", "Generated at"),
        Value = c(
          r$disease,
          sprintf("%.1f%%", 100 * r$confidence),
          if (nrow(r$ranking) >= 2) r$ranking$Disease[2] else "-",
          if (nrow(r$ranking) >= 2) sprintf("%.1f%%", 100 * r$ranking$Probability[2]) else "-",
          if (nrow(r$ranking) >= 3) r$ranking$Disease[3] else "-",
          if (nrow(r$ranking) >= 3) sprintf("%.1f%%", 100 * r$ranking$Probability[3]) else "-",
          if (is.na(r$match$score)) "-" else sprintf("%.1f%%", 100 * r$match$score),
          sprintf("%d of %d", r$match$matched, r$n_symptoms),
          r$age, r$gender, r$height, r$weight,
          if (is.na(r$bmi)) "-" else sprintf("%.1f", r$bmi),
          bmi_band(r$bmi)$label,
          r$duration, r$smoking, r$alcohol,
          if (length(r$conditions) == 0L) "None" else paste(r$conditions, collapse = "; "),
          r$n_symptoms,
          paste(unname(SYMPTOM_LABELS[r$symptoms]), collapse = "; "),
          rec_field(r$rec, "Specialist", "General Physician"),
          rec_field(r$rec, "Urgency_Level", "Consult a doctor"),
          if (isTRUE(MODEL_INFO$trained_ok)) MODEL_INFO$engine else "unavailable",
          if (isTRUE(MODEL_INFO$trained_ok)) sprintf("%.2f%%", 100 * MODEL_INFO$test_acc) else "-",
          r$time
        ),
        stringsAsFactors = FALSE
      )
      utils::write.csv(df, file, row.names = FALSE)
    }
  )

  # ---- self-contained HTML report (print it to PDF from the browser) -------
  output$download_report <- downloadHandler(
    filename = function() {
      paste0("healthcare_report_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".html")
    },
    content = function(file) {
      writeLines(build_report_html(pred_result()), file, useBytes = TRUE)
    }
  )

  # -------------------------------------------------------------------------
  # 9.9 Disease information browser  (v2 feature 8, works without predicting)
  # -------------------------------------------------------------------------
  output$browse_disease_ui <- renderUI({
    dis <- input$browse_disease %||% DISEASE_LEVELS[1]
    if (!dis %in% DISEASE_LEVELS) dis <- DISEASE_LEVELS[1]
    rec <- get_recommendation(dis)

    prof <- as.numeric(SYMPTOM_PROFILE[dis, ])
    names(prof) <- colnames(SYMPTOM_PROFILE)
    top  <- sort(prof, decreasing = TRUE)
    top  <- top[top >= MATCH_PARTIAL_CUTOFF]
    top  <- head(top, 8)

    div(
      fluidRow(
        column(6, field_block("Recommended specialist",
                 tags$span(class = "hc-badge",
                           htmltools::htmlEscape(rec_field(rec, "Specialist",
                                                           "General Physician"))))),
        column(6, field_block("Urgency level",
                 tags$b(htmltools::htmlEscape(rec_field(rec, "Urgency_Level",
                                                        "Consult a doctor")))))
      ),
      field_block("General information",
        tags$p(style = "font-size:14px; line-height:1.6; color:#34495e; margin:0;",
               htmltools::htmlEscape(rec_field(rec, "Description")))),
      div(class = "hc-divider"),
      fluidRow(
        column(6, field_block("Common symptoms",
                              bullet_list(rec_field(rec, "Common_Symptoms")))),
        column(6, field_block("Common causes",
                              bullet_list(rec_field(rec, "Causes"))))
      ),
      div(class = "hc-divider"),
      div(class = "hc-field-label",
          sprintf("Symptoms most associated with this disease in the dataset (of %s patients)",
                  format(DISEASE_COUNTS[[dis]] %||% 0, big.mark = ","))),
      div(class = "hc-chips",
        lapply(seq_along(top), function(i) {
          tags$span(class = "hc-chip",
                    sprintf("%s \u2014 %.0f%%", unname(SYMPTOM_LABELS[names(top)[i]]),
                            100 * top[i]))
        })
      ),
      div(class = "hc-divider"),
      fluidRow(
        column(6, field_block("Home care",
                              bullet_list(rec_field(rec, "Home_Care")))),
        column(6, field_block("Precautions",
                              bullet_list(rec_field(rec, "Precautions"))))
      ),
      div(class = "hc-note", style = "margin-top:4px; margin-bottom:0;",
        tags$strong("Educational information only. "),
        "Medicine information is shown for the predicted disease on the ",
        "Predict & Recommend tab. Nothing here is a prescription.")
    )
  })
}


# ============================================================================
# SECTION 10 - LAUNCH
# ============================================================================
# In RStudio simply press "Run App", or run in the console:
#     shiny::runApp()
# ----------------------------------------------------------------------------
shinyApp(ui = ui, server = server)
