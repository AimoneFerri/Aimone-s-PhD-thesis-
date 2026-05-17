# =============================================================================
# Observational Cross-Sectional Regression: Psychological / Mood Outcomes
# Generic Template — Fibre vs outcomes
# =============================================================================
#
# Design: Cross-sectional; one observation per participant at baseline.
#         Dietary fibre intake (continuous, mean-centred) is the primary
#         predictor. Each outcome is modelled independently.
#
# Model structure (two models per outcome):
#   Basic:    outcome ~ fibre_c + sex + age_c + energy_c
#   Adjusted: outcome ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c
#
# Outcomes modelled (rename to match your variables):
#   Continuous Gaussian (lm):
#     score_stress      : perceived stress (e.g. PSS total)
#     score_anxiety     : trait anxiety (e.g. STAI-T total)
#   Skewed positive continuous (lognormal / glmmTMB):
#     score_symptoms    : global symptom severity (e.g. SCL-90-R GSI)
#   Non-negative integer, zero-inflated (ZINB / glmmTMB):
#     score_depression  : depressive symptoms (e.g. BDI-II total)
#
# Each outcome section follows the same pattern:
#   1. Fit basic and adjusted models.
#   2. Run check_model() + DHARMa diagnostics on the basic model.
#   3. Export basic and adjusted models side-by-side with tab_model().
#   #3.1 Some OLS models may show residuals heteroskedasticity; consider robust estimators such as "HC-3" for corrected SEs and CIs and compare results.
#
#
# For score_symptoms and score_depression a family-selection workflow is
# included; run diagnostics and choose the model that passes before saving
# the output table.
# =============================================================================


# ---- Libraries ---------------------------------------------------------------

library(readr)
library(dplyr)
library(tidyr)
library(summarytools)
library(ggmice)
library(ggplot2)
library(GGally)
library(viridis)
library(gtsummary)
library(flextable)
library(glmmTMB)
library(easystats)
library(sjPlot)
library(DHARMa)

options(scipen = 999)


# ---- Study constants ---------------------------------------------------------
# Adapt fibre grouping levels / labels and employment reference to your dataset.
# These constants are used for the descriptive table and factor coding only;
# models use the continuous fibre predictor.

fibre_group_levels <- c("LF", "MF", "HF")
fibre_group_labels <- c("Low Fibre", "Moderate Fibre", "High Fibre")

# Reference level for employment_status factor (adapt to your largest / most
# natural reference group; remaining levels are appended in the order listed)
employment_levels <- c("Full-time", "Part-time", "Student", "Unemployed/Other")

# Outcome variable names as they appear in the input file
outcome_vars <- c("score_stress", "score_anxiety", "score_depression", "score_symptoms")

# Variables that must be non-missing for complete-case selection
primary_vars  <- c(outcome_vars, "fibre_intake", "age", "sex", "energy_kcal")
adjusted_vars <- c(primary_vars, "employment_status", "pal")


# ---- Load data ---------------------------------------------------------------
# Required columns (adapt names to match your dataset):
#   id              — participant identifier
#   fibre_grouping  — categorical fibre group (LF / MF / HF)
#   fibre_intake    — dietary fibre (g/day, continuous)
#   age             — participant age (years)
#   sex             — biological sex (factor)
#   energy_kcal     — total energy intake (kcal/day)
#   employment_status
#   pal             — physical activity level (continuous index or MET)
#   sleep_quality   — sleep quality score (e.g. PSQI global; higher = worse)
#   score_stress, score_anxiety, score_depression, score_symptoms

obs_data <- read_csv("data/derived/YOUR_FILE.csv") %>%
  mutate(id = as.factor(id))


# ---- Inspect data ------------------------------------------------------------

obs_data %>%
  select(all_of(outcome_vars)) %>%
  dfSummary() %>%
  stview()


# ---- Missing data pattern ----------------------------------------------------

obs_data %>%
  select(all_of(primary_vars)) %>%
  plot_pattern()


# ---- EDA: correlation matrix -------------------------------------------------

# Sleep quality is included in the EDA to visualise its associations with fibre
# and mood outcomes. It is not included in the regression models because it
# likely acts as a mediator on the fibre → mood pathway; adjusting for a
# mediator would attenuate or block the effect of interest.

eda_mood <- obs_data %>%
  ggpairs(
    columns      = c("fibre_intake", "sleep_quality", "score_stress",
                     "score_anxiety", "score_depression", "score_symptoms"),
    columnLabels = c("Fibre (g/day)", "Sleep Quality", "Stress",
                     "Anxiety", "Depression", "Global Symptoms"),
    lower        = list(continuous = "smooth")
  )

ggsave("plots/eda_mood.png", eda_mood, width = 12, height = 10, dpi = 300)

# Grouped version coloured by fibre intake category
eda_mood_grouped <- obs_data %>%
  mutate(fibre_grouping = factor(fibre_grouping,
                                 levels = fibre_group_levels,
                                 labels = fibre_group_labels)) %>%
  ggpairs(
    mapping      = aes(colour = fibre_grouping),
    columns      = c("fibre_intake", "sleep_quality", "score_stress",
                     "score_anxiety", "score_depression", "score_symptoms"),
    columnLabels = c("Fibre (g/day)", "Sleep Quality", "Stress",
                     "Anxiety", "Depression", "Global Symptoms"),
    lower        = list(continuous = "smooth")
  ) +
  scale_colour_viridis_d(option = "cividis", begin = 0.2, end = 0.8) +
  scale_fill_viridis_d(option = "cividis", begin = 0.2, end = 0.8)

ggsave("plots/eda_mood_grouped.png", eda_mood_grouped, width = 12, height = 10, dpi = 300)


# ---- Descriptive table -------------------------------------------------------
# Outcomes are right-skewed; Kruskal-Wallis used for unadjusted group comparison. Descriptives purposes only and to detect possible non-monotonic relationships between recruited groups.

mood_desc_table <- obs_data %>%
  mutate(fibre_grouping = factor(fibre_grouping,
                                 levels = fibre_group_levels,
                                 labels = fibre_group_labels)) %>%
  select(fibre_grouping, all_of(outcome_vars)) %>%
  tbl_summary(
    by        = fibre_grouping,
    statistic = list(all_continuous() ~ "{median} ({p25}, {p75})"),
    label     = list(
      score_stress     ~ "Perceived Stress",
      score_anxiety    ~ "Trait Anxiety",
      score_depression ~ "Depressive Symptoms",
      score_symptoms   ~ "Global Symptom Severity"
    ),
    missing = "no"
  ) %>%
  add_overall(col_label = "**Total**") %>%
  add_p(test = everything() ~ "kruskal.test") %>%
  modify_header(
    stat_1 ~ "**Low Fibre**",
    stat_2 ~ "**Moderate Fibre**",
    stat_3 ~ "**High Fibre**"
  ) %>%
  modify_footnote(
    all_stat_cols() ~ "Median (IQR)",
    p.value         ~ "Unadjusted p-values from Kruskal-Wallis test"
  ) %>%
  bold_labels()

mood_desc_table %>%
  as_flex_table() %>%
  save_as_docx(path = "tables/OBS_mood_descriptives.docx")


# ---- Data preparation --------------------------------------------------------

# Complete-case dataset (minimum variables for basic models)
obs_cc <- obs_data %>%
  drop_na(all_of(primary_vars))

# Mean-centre continuous predictors and covariates.
# scale = FALSE centres without dividing by SD, preserving original units so
# that model coefficients reflect change per unit of the original variable.
obs_cc <- obs_cc %>%
  mutate(
    fibre_c           = scale(fibre_intake,  center = TRUE, scale = FALSE),
    age_c             = scale(age,           center = TRUE, scale = FALSE),
    energy_c          = scale(energy_kcal,   center = TRUE, scale = FALSE),
    pal_c             = scale(pal,           center = TRUE, scale = FALSE),
    sex               = as.factor(sex),
    employment_status = factor(employment_status, levels = employment_levels)
  )

# Save model-ready dataset (dated for reproducibility)
write.csv(
  obs_cc,
  file      = paste0("data/derived/obs_mood_data_", Sys.Date(), ".csv"),
  row.names = FALSE
)


# =============================================================================
# REGRESSION MODELS
# =============================================================================
# Structure:
#   Basic:    outcome ~ fibre_c + sex + age_c + energy_c
#   Adjusted: outcome ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c
#
# For each outcome:
#   1. Fit basic and adjusted models.
#   2. Run check_model() and DHARMa diagnostics on the basic model.
#      (If the basic model fits well the adjusted will too; re-run diagnostics
#       on the adjusted model if there is any reason to suspect otherwise.)
#   3. Export basic and adjusted side-by-side with tab_model().
# =============================================================================


# ---- Perceived Stress (Gaussian lm) -----------------------------------------

mod_stress <- lm(
  score_stress ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_stress_adj <- lm(
  score_stress ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_stress)
res_stress <- simulateResiduals(mod_stress, n = 1000)
testResiduals(res_stress)

tab_model(
  mod_stress, mod_stress_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_stress.doc"
)


# ---- Trait Anxiety (Gaussian lm) ---------------------------------------------

mod_anxiety <- lm(
  score_anxiety ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_anxiety_adj <- lm(
  score_anxiety ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_anxiety)
res_anxiety <- simulateResiduals(mod_anxiety, n = 1000)
testResiduals(res_anxiety)

tab_model(
  mod_anxiety, mod_anxiety_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_anxiety.doc"
)


# ---- Global Symptom Severity (lognormal / glmmTMB) --------------------------
# GSI is a continuous positive score with strong right skew; a log-link model
# is preferred over Gaussian lm. Lognormal and Gamma(link = "log") are compared
# below via AIC — select whichever shows better DHARMa residuals.
# Exponentiated estimates are multiplicative effects on the geometric mean.

mod_symptoms_log <- glmmTMB(
  score_symptoms ~ fibre_c + sex + age_c + energy_c,
  family = lognormal(link = "log"),
  data   = obs_cc
)
mod_symptoms_log_adj <- glmmTMB(
  score_symptoms ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  family = lognormal(link = "log"),
  data   = obs_cc
)

# Gamma alternative — compare to lognormal
mod_symptoms_gamma <- glmmTMB(
  score_symptoms ~ fibre_c + sex + age_c + energy_c,
  family = Gamma(link = "log"),
  data   = obs_cc
)
compare_performance(mod_symptoms_log, mod_symptoms_gamma)

# Diagnostics for preferred model; replace object name if Gamma is selected
res_symptoms <- simulateResiduals(mod_symptoms_log, n = 1000)
testResiduals(res_symptoms)

# Replace mod_symptoms_log / mod_symptoms_log_adj with the Gamma objects if
# compare_performance() favours Gamma and diagnostics confirm the choice
tab_model(
  mod_symptoms_log, mod_symptoms_log_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_symptoms.doc"
)


# ---- Depressive Symptoms (zero-inflated negative binomial / glmmTMB) --------
# Depression questionnaires (e.g. BDI-II) produce non-negative integer scores
# with right skew and frequent zero values. Recommended workflow:
#   Step 1 — Fit Gaussian lm as a diagnostic baseline. Residuals typically show
#             non-normality and excess zeros, motivating a count model.
#   Step 2 — Fit negative binomial (nbinom2: quadratic overdispersion) via
#             glmmTMB. testZeroInflation() determines whether a zero-inflation
#             component (ziformula = ~1) is necessary.
#   Step 3 — Confirm final model with DHARMa diagnostics.
#
# Sleep quality is not included here despite its known association with
# depression, because it likely mediates the fibre → depression pathway.
# Its relationship with fibre and outcomes is explored in the EDA instead.

# Step 1 — Gaussian baseline check
mod_depression_lm <- lm(
  score_depression ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
check_model(mod_depression_lm)
res_dep_lm <- simulateResiduals(mod_depression_lm, n = 1000)
plot(res_dep_lm)
testZeroInflation(res_dep_lm)

# Step 2 — Negative binomial with zero-inflation component
# Remove ziformula = ~1 if testZeroInflation() above is non-significant
mod_depression <- glmmTMB(
  score_depression ~ fibre_c + sex + age_c + energy_c,
  ziformula = ~1,
  family    = nbinom2,
  data      = obs_cc
)
mod_depression_adj <- glmmTMB(
  score_depression ~ fibre_c + sex + age_c + energy_c +
    employment_status + pal_c,
  ziformula = ~1,
  family    = nbinom2,
  data      = obs_cc
)

# Step 3 — Diagnostics
check_zeroinflation(mod_depression)
check_overdispersion(mod_depression)
res_dep <- simulateResiduals(mod_depression, n = 1000)
testResiduals(res_dep)

tab_model(
  mod_depression, mod_depression_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_depression.doc"
)
