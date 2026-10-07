# =============================================================================
#  generate_demo_data.R
#  ---------------------------------------------------------------------------
#  OPTIONAL UTILITY SCRIPT - NOT required to run the application.
#
#  This script creates the DEMO datasets used by the app:
#       healthcare_data.csv   (synthetic patient records)
#       recommendations.csv   (disease -> specialist / advice lookup table)
#
#  The data is SYNTHETIC (computer generated from probability profiles).
#  It is NOT real patient data and must NOT be used for real medical purposes.
#
#  >>> REPLACING WITH A REAL DATASET <<<
#  If you obtain a real dataset, simply overwrite healthcare_data.csv keeping
#  the same column names:
#       PatientID, Age, Gender, <28 symptom columns as 0/1>, Disease
#  Then update SYMPTOM_COLS in app.R if your symptom column names differ.
#  You do NOT need to run this script again.
#
#  Run with:  Rscript generate_demo_data.R
# =============================================================================

set.seed(2024)

# ---------------------------------------------------------------------------
# 1. Symptom column names  (0 = symptom absent, 1 = symptom present)
# ---------------------------------------------------------------------------
SYMPTOM_COLS <- c(
  "fever", "cough", "headache", "fatigue", "nausea", "vomiting", "sneezing",
  "sore_throat", "body_pain", "stomach_pain", "breathing_difficulty",
  "runny_nose", "chills", "joint_pain", "skin_rash", "chest_pain",
  "dizziness", "loss_of_appetite", "diarrhea", "constipation",
  "burning_urination", "blurred_vision", "night_sweats", "weight_loss",
  "yellow_eyes", "palpitations", "frequent_urination", "excessive_thirst"
)

# ---------------------------------------------------------------------------
# 2. Disease profiles
#    core   = the symptoms typical of that disease, with the probability that
#             a patient with that disease shows the symptom
#    age    = mean & sd of patient age for that disease (gives the model a
#             realistic age signal), truncated to [lo, hi]
#    female = probability the record is Female
# ---------------------------------------------------------------------------
profiles <- list(
  "Common Cold" = list(
    core = c(sneezing = .90, runny_nose = .90, sore_throat = .80, cough = .75,
             headache = .50, fatigue = .50, fever = .25),
    age = c(mean = 28, sd = 14, lo = 3, hi = 65), female = .50
  ),
  "Influenza" = list(
    core = c(fever = .92, cough = .80, body_pain = .85, headache = .75,
             fatigue = .85, chills = .80, sore_throat = .60, joint_pain = .60),
    age = c(mean = 30, sd = 15, lo = 4, hi = 70), female = .50
  ),
  "Viral Fever" = list(
    core = c(fever = .95, headache = .80, body_pain = .80, fatigue = .80,
             nausea = .50, chills = .60, loss_of_appetite = .60),
    age = c(mean = 31, sd = 16, lo = 3, hi = 75), female = .50
  ),
  "Dengue" = list(
    core = c(fever = .97, headache = .85, joint_pain = .85, body_pain = .85,
             skin_rash = .70, nausea = .60, fatigue = .80, vomiting = .45,
             loss_of_appetite = .60),
    age = c(mean = 32, sd = 14, lo = 4, hi = 70), female = .48
  ),
  "Malaria" = list(
    core = c(fever = .96, chills = .90, headache = .80, vomiting = .60,
             fatigue = .80, nausea = .60, body_pain = .70,
             loss_of_appetite = .60),
    age = c(mean = 34, sd = 16, lo = 3, hi = 75), female = .48
  ),
  "Typhoid" = list(
    core = c(fever = .95, stomach_pain = .75, headache = .70, fatigue = .80,
             loss_of_appetite = .75, constipation = .50, diarrhea = .40,
             vomiting = .45),
    age = c(mean = 27, sd = 14, lo = 3, hi = 65), female = .48
  ),
  "Pneumonia" = list(
    core = c(fever = .85, cough = .92, breathing_difficulty = .85,
             chest_pain = .70, fatigue = .80, chills = .70,
             loss_of_appetite = .55),
    age = c(mean = 45, sd = 20, lo = 3, hi = 88), female = .50
  ),
  "Bronchial Asthma" = list(
    core = c(breathing_difficulty = .95, cough = .85, chest_pain = .60,
             fatigue = .60, sneezing = .40, runny_nose = .35),
    age = c(mean = 24, sd = 14, lo = 3, hi = 62), female = .52
  ),
  "Tuberculosis" = list(
    core = c(cough = .95, night_sweats = .85, weight_loss = .85, fever = .75,
             fatigue = .80, loss_of_appetite = .70, chest_pain = .50,
             breathing_difficulty = .45),
    age = c(mean = 41, sd = 16, lo = 8, hi = 80), female = .42
  ),
  "Gastroenteritis" = list(
    core = c(diarrhea = .90, vomiting = .85, stomach_pain = .85, nausea = .85,
             fever = .50, fatigue = .65, loss_of_appetite = .60),
    age = c(mean = 29, sd = 17, lo = 2, hi = 78), female = .50
  ),
  "Food Poisoning" = list(
    core = c(nausea = .90, vomiting = .90, diarrhea = .85, stomach_pain = .85,
             fever = .40, fatigue = .60, dizziness = .40),
    age = c(mean = 27, sd = 13, lo = 4, hi = 68), female = .50
  ),
  "Migraine" = list(
    core = c(headache = .97, nausea = .70, dizziness = .60, blurred_vision = .50,
             vomiting = .45, fatigue = .50),
    age = c(mean = 33, sd = 13, lo = 10, hi = 66), female = .68
  ),
  "Hypertension" = list(
    core = c(headache = .70, dizziness = .65, palpitations = .60, chest_pain = .50,
             fatigue = .55, blurred_vision = .45, breathing_difficulty = .30),
    age = c(mean = 57, sd = 10, lo = 38, hi = 88), female = .48
  ),
  "Diabetes Mellitus" = list(
    core = c(frequent_urination = .90, excessive_thirst = .90, fatigue = .80,
             weight_loss = .60, blurred_vision = .60, dizziness = .40),
    age = c(mean = 54, sd = 11, lo = 30, hi = 86), female = .50
  ),
  "Allergic Rhinitis" = list(
    core = c(sneezing = .95, runny_nose = .90, skin_rash = .55, cough = .50,
             headache = .45, fatigue = .40, sore_throat = .40),
    age = c(mean = 25, sd = 12, lo = 3, hi = 58), female = .50
  ),
  "Urinary Tract Infection" = list(
    core = c(burning_urination = .95, stomach_pain = .70, fever = .60,
             fatigue = .60, nausea = .40, vomiting = .30),
    age = c(mean = 38, sd = 16, lo = 14, hi = 80), female = .78
  ),
  "Anemia" = list(
    core = c(fatigue = .95, dizziness = .80, headache = .60, palpitations = .60,
             breathing_difficulty = .50, loss_of_appetite = .50),
    age = c(mean = 31, sd = 15, lo = 8, hi = 72), female = .75
  ),
  "Hepatitis" = list(
    core = c(yellow_eyes = .90, loss_of_appetite = .80, nausea = .75,
             vomiting = .65, fatigue = .85, stomach_pain = .60, fever = .60,
             joint_pain = .40),
    age = c(mean = 36, sd = 14, lo = 8, hi = 72), female = .45
  )
)

# ---------------------------------------------------------------------------
# 3. Helper: truncated normal age
# ---------------------------------------------------------------------------
rtrunc_age <- function(n, mean, sd, lo, hi) {
  round(pmin(pmax(stats::rnorm(n, mean, sd), lo), hi))
}

# ---------------------------------------------------------------------------
# 4. Generate the records
# ---------------------------------------------------------------------------
BACKGROUND_RATE <- 0.04   # chance a "non-typical" symptom still appears

# Records per disease - deliberately NOT equal, so that the disease
# distribution chart looks like real-world prevalence data.
N_BY_DISEASE <- c(
  "Common Cold"             = 130, "Influenza"            = 115,
  "Viral Fever"             = 100, "Dengue"               =  95,
  "Malaria"                 =  85, "Typhoid"              =  70,
  "Pneumonia"               =  75, "Bronchial Asthma"     =  90,
  "Tuberculosis"            =  65, "Gastroenteritis"      = 105,
  "Food Poisoning"          =  88, "Migraine"             =  92,
  "Hypertension"            =  78, "Diabetes Mellitus"    =  82,
  "Allergic Rhinitis"       =  86, "Urinary Tract Infection" = 74,
  "Anemia"                  =  68, "Hepatitis"            =  60
)

records <- vector("list", length(profiles))
names(records) <- names(profiles)

for (d in names(profiles)) {
  p    <- profiles[[d]]
  core <- p$core
  n    <- as.integer(N_BY_DISEASE[[d]])

  # --- symptoms -------------------------------------------------------------
  sym_mat <- matrix(0L, nrow = n, ncol = length(SYMPTOM_COLS),
                    dimnames = list(NULL, SYMPTOM_COLS))

  # background noise for EVERY symptom
  sym_mat[] <- stats::rbinom(n * length(SYMPTOM_COLS), 1, BACKGROUND_RATE)

  # typical symptoms for this disease
  for (s in names(core)) {
    sym_mat[, s] <- stats::rbinom(n, 1, core[[s]])
  }

  # guarantee at least 2 typical symptoms per record (otherwise the record
  # carries no learnable signal at all)
  top2 <- names(sort(core, decreasing = TRUE))[1:2]
  weak <- rowSums(sym_mat[, names(core), drop = FALSE]) < 2
  if (any(weak)) {
    for (s in top2) sym_mat[weak, s] <- 1L
  }

  # --- demographics ---------------------------------------------------------
  a  <- p$age
  age_vec <- rtrunc_age(n, a[["mean"]], a[["sd"]], a[["lo"]], a[["hi"]])
  gender  <- ifelse(stats::runif(n) < p$female, "Female", "Male")

  records[[d]] <- data.frame(
    Age    = age_vec,
    Gender = gender,
    sym_mat,
    Disease = d,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

healthcare_data <- do.call(rbind, records)
healthcare_data <- healthcare_data[sample(nrow(healthcare_data)), ]   # shuffle
healthcare_data$PatientID <- sprintf("P%04d", seq_len(nrow(healthcare_data)))
healthcare_data <- healthcare_data[, c("PatientID", "Age", "Gender",
                                       SYMPTOM_COLS, "Disease")]
rownames(healthcare_data) <- NULL

write.csv(healthcare_data, "healthcare_data.csv", row.names = FALSE)

cat("healthcare_data.csv written:",
    nrow(healthcare_data), "rows x", ncol(healthcare_data), "columns\n")

# ---------------------------------------------------------------------------
# 5. Recommendation lookup table
#    NOTE: "Medicines_Educational" is general, publicly documented information
#          for learning purposes only. It is NOT a prescription and is NOT
#          medical advice.
# ---------------------------------------------------------------------------
recs <- data.frame(
  Disease = c(
    "Common Cold", "Influenza", "Viral Fever", "Dengue", "Malaria", "Typhoid",
    "Pneumonia", "Bronchial Asthma", "Tuberculosis", "Gastroenteritis",
    "Food Poisoning", "Migraine", "Hypertension", "Diabetes Mellitus",
    "Allergic Rhinitis", "Urinary Tract Infection", "Anemia", "Hepatitis"
  ),
  Specialist = c(
    "General Physician", "General Physician", "General Physician",
    "General Physician / Infectious Disease Specialist", "General Physician",
    "General Physician", "Pulmonologist", "Pulmonologist",
    "Pulmonologist / Chest Specialist", "Gastroenterologist",
    "Gastroenterologist", "Neurologist", "Cardiologist", "Endocrinologist",
    "ENT Specialist / Allergist", "Urologist", "Hematologist",
    "Gastroenterologist / Hepatologist"
  ),
  Description = c(
    "A mild viral infection of the nose and throat. Usually settles on its own within 7-10 days. Antibiotics do not help because the cause is a virus.",
    "A viral infection with sudden high fever, body ache and chills. Spreads easily, so rest and isolation help. Annual vaccination reduces risk.",
    "A nonspecific fever caused by a viral infection. The fever usually lasts 3-5 days and improves with rest, fluids and fever control.",
    "A mosquito-borne viral infection. Fever is often very high and is accompanied by severe joint and muscle pain and sometimes a skin rash. Some cases become severe, so monitoring platelet counts is important.",
    "A mosquito-borne parasitic infection. It characteristically causes recurring high fever with severe chills. It needs a blood test for confirmation and specific treatment.",
    "A bacterial infection of the gut spread through contaminated food or water. It causes a long, stepwise rising fever along with abdominal symptoms.",
    "A lung infection in which the air sacs fill with fluid. It causes a productive cough and difficulty in breathing, and needs prompt medical treatment.",
    "A long-term inflammatory condition of the airways. Symptoms come in episodes triggered by dust, pollen, cold air or exercise. Inhalers are the mainstay of control.",
    "A bacterial infection that mainly affects the lungs and spreads through the air. It requires a long course of treatment and is a public-health priority case.",
    "Inflammation of the stomach and intestines, usually from an infection. The main risk is dehydration, so fluids and salts matter most.",
    "Illness caused by eating food contaminated with bacteria or their toxins. It starts within hours of the meal, with vomiting and loose motions.",
    "A recurring type of headache, often one-sided and throbbing, sometimes with nausea and sensitivity to light and sound. Triggers vary from person to person.",
    "Persistently raised blood pressure. It usually has no symptoms for years but silently damages the heart, brain, kidneys and eyes.",
    "A long-term condition in which blood sugar stays high because the body does not make or properly use insulin. Needs lifelong lifestyle care and monitoring.",
    "An allergic reaction of the nose and eyes to dust, pollen or animal hair. It is not infectious and often runs in families.",
    "An infection of the bladder or urinary tract, commonly caused by bacteria entering from outside. Very common in women.",
    "A shortage of healthy red blood cells or haemoglobin. The commonest cause in India is iron deficiency. It causes tiredness and breathlessness.",
    "Inflammation of the liver, most often caused by a virus. The liver may not work normally, which shows up as yellow eyes and skin."
  ),
  Medicines_Educational = c(
    "Paracetamol (fever/pain); Cetirizine (runny nose); Saline nasal drops; Steam inhalation. Antibiotics are NOT useful for a viral cold.",
    "Paracetamol; Oseltamivir (only if a doctor prescribes it early); Cetirizine for sneezing. Bed rest and plenty of fluids.",
    "Paracetamol for fever and body ache; Oral rehydration salts; Sponging with lukewarm water for high fever.",
    "Paracetamol for fever and pain. AVOID aspirin and ibuprofen-type painkillers (NSAIDs) unless a doctor approves them, because of bleeding risk. Hospital monitoring if warning signs appear.",
    "Antimalarial drugs (e.g. chloroquine/artemisinin combination therapy) prescribed by a doctor after a positive blood test; Paracetamol for fever.",
    "A full course of antibiotics (e.g. azithromycin, cefixime) as prescribed by a doctor. Never stop a course midway.",
    "Antibiotics as prescribed; Paracetamol for fever; Nebulisation or oxygen if breathing is difficult. This usually needs hospital care.",
    "Salbutamol inhaler for sudden attacks; Inhaled corticosteroid (e.g. budesonide) for long-term control; Montelukast. Inhalers are safe and not habit-forming.",
    "A fixed combination of anti-TB drugs (isoniazid, rifampicin, ethambutol, pyrazinamide) taken daily for 6 months or more, free of cost at government DOTS centres.",
    "Oral rehydration salts (ORS); Zinc supplements; Antibiotics only if a bacterial cause is confirmed. Avoid anti-diarrhoeal medicines without medical advice.",
    "Oral rehydration salts; Plenty of fluids; Paracetamol if there is fever. Vomiting usually settles in 24-48 hours.",
    "Paracetamol or a doctor-prescribed painkiller; An anti-emetic (e.g. domperidone) for nausea; Rest in a dark, quiet room.",
    "Blood-pressure medicines are lifelong and must only be started or changed by a doctor, e.g. amlodipine, telmisartan, hydrochlorothiazide. Lifestyle changes reduce the dose needed.",
    "Metformin is the usual first medicine; Insulin for some patients. Diet, exercise and regular blood-sugar testing are as important as tablets.",
    "Non-sedating antihistamines (e.g. loratadine, cetirizine); Saline nasal rinse; Steroid nasal spray for persistent symptoms.",
    "Antibiotics as prescribed; Paracetamol for fever and pain; Drink 2-3 litres of water a day.",
    "Iron and folic acid tablets (usually taken with vitamin C); Vitamin B12 if deficient. Always confirm the type of anaemia with a blood test first.",
    "Rest; A high-carbohydrate, low-fat diet; Vitamin supplements. Hepatitis A and E usually recover with supportive care; B and C need specialist treatment."
  ),
  Home_Care = c(
    "Warm fluids, adequate sleep, steam inhalation, avoid cold drinks.",
    "Bed rest, warm soups, isolate from family members, use a mask.",
    "Plenty of water and ORS, light easily digestible food, complete rest.",
    "Strict bed rest, 3-4 litres of fluids per day, monitor temperature every 4-6 hours, watch for bleeding gums or nose bleeds.",
    "Complete rest, high fluid intake, use a mosquito net, take medicines at fixed times.",
    "Soft, bland, low-fibre food; boiled or bottled water; complete rest.",
    "Rest in a semi-sitting position, warm fluids, follow up urgently if breathing worsens.",
    "Avoid dust, smoke and cold air; keep the inhaler with you at all times; do breathing exercises.",
    "High-protein, high-calorie diet; cover the mouth while coughing; ventilate the room; never skip doses.",
    "ORS after every loose motion, bland food such as rice and curd, avoid oily and spicy food.",
    "Stop eating solid food for a few hours, sip ORS slowly and frequently.",
    "Rest in a dark quiet room, cold compress on the forehead, note down what triggered the attack.",
    "Reduce salt, walk 30 minutes daily, stop smoking, reduce weight, check BP twice a week.",
    "Sugar-free diet, daily exercise, regular foot care, monitor blood sugar as advised.",
    "Keep the house dust-free, wash bedding in hot water, avoid pets during peak season.",
    "Drink plenty of water, do not hold urine, maintain personal hygiene.",
    "Iron-rich food: green leafy vegetables, dates, jaggery, liver; avoid tea or coffee with meals.",
    "Complete rest, no alcohol at all, low-fat food, drink boiled water."
  ),
  # --- NEW in v2: used by the "Disease information card" -------------------
  Common_Symptoms = c(
    "Sneezing; Runny nose; Sore throat; Mild cough; Mild fever; Headache",
    "Sudden high fever; Body and muscle ache; Dry cough; Headache; Chills; Tiredness",
    "Fever; Headache; Body pain; Tiredness; Loss of appetite; Nausea",
    "Very high fever; Severe joint and muscle pain; Skin rash; Headache; Nausea; Pain behind the eyes",
    "High fever with shivering chills; Headache; Vomiting; Tiredness; Heavy sweating",
    "Fever that rises step by step over days; Stomach pain; Headache; Constipation or loose motions; Weakness; Loss of appetite",
    "Cough with phlegm; Difficulty in breathing; Fever with chills; Chest pain; Tiredness",
    "Episodes of difficulty in breathing; Wheezing; Dry cough, worse at night; Chest tightness",
    "Cough lasting more than 2 weeks; Blood in sputum; Night sweats; Weight loss; Evening fever; Tiredness",
    "Loose motions; Vomiting; Stomach cramps; Nausea; Mild fever",
    "Vomiting starting within hours of eating; Loose motions; Stomach cramps; Nausea; Dizziness",
    "One-sided throbbing headache; Nausea; Vomiting; Sensitivity to light and sound; Blurred vision",
    "Usually no symptoms at all; Headache; Dizziness; Palpitations; Blurred vision",
    "Frequent urination; Excessive thirst; Tiredness; Weight loss; Blurred vision; Slow-healing wounds",
    "Frequent sneezing; Runny nose; Itchy watery eyes; Blocked nose; Skin rash",
    "Burning while passing urine; Frequent urge to urinate; Lower abdominal pain; Fever; Cloudy or foul-smelling urine",
    "Tiredness; Dizziness; Shortness of breath; Pale skin; Palpitations; Headache",
    "Yellow eyes and skin; Loss of appetite; Nausea and vomiting; Tiredness; Dark urine; Stomach pain"
  ),
  Causes = c(
    "More than 200 viruses, most often rhinovirus. Spreads through droplets from coughs and sneezes, or by touching contaminated surfaces and then the face.",
    "Influenza A and B viruses. Spreads mainly through airborne droplets. New strains appear every year, which is why the vaccine is repeated annually.",
    "A group of common viral infections rather than one specific virus. Spreads through droplets or contaminated food and water.",
    "Dengue virus, spread by the bite of an infected Aedes aegypti mosquito. These mosquitoes breed in clean stagnant water and usually bite during the day.",
    "Plasmodium parasites, spread by the bite of an infected female Anopheles mosquito, which bites mainly between dusk and dawn.",
    "Salmonella typhi bacteria, spread through food or water contaminated by the faeces of an infected person.",
    "Bacteria (most commonly Streptococcus pneumoniae), viruses or fungi infect the air sacs of the lungs, which then fill with fluid.",
    "Long-term inflammation of the airways with a genetic tendency. Attacks are triggered by dust, pollen, cold air, smoke or exercise. It is not infectious.",
    "Mycobacterium tuberculosis bacteria, spread through the air when an infected person coughs or sneezes. Spreads fastest in crowded, poorly ventilated places.",
    "Viruses (such as rotavirus and norovirus), bacteria or parasites, spread through contaminated food, water or hands.",
    "Toxins or bacteria (Staphylococcus, Salmonella, E. coli) in food that was stale, undercooked or left unrefrigerated.",
    "A neurological condition with a genetic tendency. Attacks are triggered by stress, lack of sleep, skipped meals, strong smells or certain foods. It is NOT caused by a brain tumour.",
    "Persistently raised blood pressure. Contributing causes include excess salt, obesity, lack of exercise, stress, alcohol and heredity. In most patients no single cause is found.",
    "The body either does not make enough insulin (type 1) or cannot use it properly (type 2). Type 2 is strongly linked to obesity, inactivity and family history.",
    "An allergic reaction to dust mites, pollen, animal hair or mould. It is not an infection and does not spread from person to person.",
    "Bacteria, usually E. coli, entering the urinary tract. More common in women because the urethra is shorter. Holding urine and poor hygiene raise the risk.",
    "Too little haemoglobin or too few red blood cells. In India the commonest cause is iron deficiency; others are vitamin B12 deficiency, blood loss and long-term disease.",
    "Usually a viral infection (hepatitis A, B, C, D or E). A and E spread through contaminated food and water; B and C spread through infected blood, needles and unprotected sex. Alcohol and some drugs also damage the liver."
  ),
  Precautions = c(
    "See a doctor if symptoms last more than 10 days, or if there is high fever or ear pain.",
    "Seek help if breathing becomes difficult, or if fever lasts beyond 5 days.",
    "See a doctor if fever crosses 103 F, lasts more than 3 days, or if there is a rash or bleeding.",
    "GO TO HOSPITAL IMMEDIATELY if there is severe abdominal pain, persistent vomiting, bleeding gums, black stools or cold clammy skin.",
    "Get a blood test immediately; do not self-medicate; return if vomiting prevents you from taking fluids.",
    "See a doctor for a blood culture test; untreated typhoid can cause intestinal perforation.",
    "MEDICAL EMERGENCY if lips turn blue, breathing is very fast, or there is confusion. Go to hospital now.",
    "Go to hospital if the inhaler does not relieve an attack within 15 minutes.",
    "Do not share food or utensils; complete the entire course; report weight loss or blood in sputum.",
    "See a doctor if there is blood in the stools, no urine for 8 hours, or signs of severe dehydration.",
    "See a doctor if symptoms last more than 2 days or there is blood in vomit or stools.",
    "See a doctor if the headache is sudden and severe, or comes with weakness or slurred speech.",
    "See a doctor immediately for chest pain, breathlessness, or a very severe headache. Do not stop medicines on your own.",
    "See a doctor urgently if there is confusion, very high sugar, deep rapid breathing or a non-healing wound.",
    "See a doctor if there is wheezing or breathlessness.",
    "See a doctor if there is back pain with high fever, or blood in the urine. Men with these symptoms need prompt review.",
    "See a doctor if there is chest pain, breathlessness at rest, or fainting.",
    "See a doctor urgently if drowsiness, confusion or bleeding develops. Avoid alcohol and self-medication completely."
  ),
  Urgency_Level = c(
    "Routine", "See a doctor in 1-2 days", "See a doctor in 1-2 days",
    "Urgent - same day", "Urgent - same day", "See a doctor in 1-2 days",
    "Emergency - go now", "See a doctor in 1-2 days", "See a doctor soon",
    "See a doctor in 1-2 days", "Routine", "Routine",
    "See a doctor soon", "See a doctor soon", "Routine",
    "See a doctor in 1-2 days", "See a doctor soon", "Urgent - same day"
  ),
  stringsAsFactors = FALSE
)

write.csv(recs, "recommendations.csv", row.names = FALSE)
cat("recommendations.csv written:", nrow(recs), "rows x", ncol(recs), "columns\n")
