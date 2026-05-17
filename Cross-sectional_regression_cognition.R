# =============================================================================
# Observational Cross-Sectional Regression: Cognitive Outcomes
# Generic Template — Fibre vs outcomes
# =============================================================================

#         Dietary fibre intake (continuous, mean-centred) is the primary
#         predictor. Each outcome is modelled independently.
#
# Model structure (two models per outcome):
#   Basic:    outcome ~ fibre_c + sex + age_c + energy_c
#   Adjusted: outcome ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c
#   adjsutment for energy needed for confounding effect of increased overall food intake, age and sex explain cognitive outcomes, while physical activity level (PAL) and employment status are additional key confounders needed to estimate the overall effect of fibre. See/use DAG.
#
# Outcomes modelled (rename to match your variables):
#   Continuous Gaussian (lm):
#     score_memory_total     : verbal learning total score
#     score_memory_immediate : immediate recall
#     score_memory_delayed   : delayed recall
#     score_wm_errors        : working memory between-errors
#     score_wm_strategy      : working memory strategy score
#     score_set_shifting     : set-shifting adjusted errors (log-transformed)
#     score_attention        : sustained attention probability
#     score_emotion_recog    : emotion recognition total hits
#     score_attentional_bias : emotional attentional bias score
#     score_decision_making  : decision-making total performance
#     score_inhibition       : inhibitory control interference effect
#   Bounded proportion (0–100 %) — beta-binomial glmmTMB (see dedicated section):
#     score_recog_memory_imm : pattern recognition memory immediate (%)
#     score_recog_memory_del : pattern recognition memory delayed (%)
#
# Each outcome section follows the same pattern:
#   1. Fit basic and adjusted models.
#   2. Run check_model() + DHARMa diagnostics on the basic model.
#   3. Export basic and adjusted models side-by-side with tab_model().
#   3.1 Some OLS models may show residuals heteroskedasticity; consider robust estimators such as "HC-3" for corrected SEs and CIs and compare results.
#
# For score_set_shifting a log transformation is applied to the outcome in the
# model formula. For recognition memory outcomes a beta-binomial workflow is
# used (Step 1 Gaussian baseline check, Step 2 beta-binomial).
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

fibre_group_levels <- c("LF", "MF", "HF")
fibre_group_labels <- c("Low Fibre", "Moderate Fibre", "High Fibre")

# Reference level for employment_status factor
employment_levels <- c("Full-time", "Part-time", "Student", "Unemployed/Other")

# Number of test trials for proportion outcomes (adapt to your paradigm)
prop_trials <- 18

# Continuous outcome names as they appear in the input file
continuous_outcomes <- c(
  "score_memory_total",
  "score_memory_immediate",
  "score_memory_delayed",
  "score_wm_errors",
  "score_wm_strategy",
  "score_set_shifting",
  "score_attention",
  "score_emotion_recog",
  "score_attentional_bias",
  "score_decision_making",
  "score_inhibition"
)

# Proportion outcomes (0–100 %); modelled separately with beta-binomial
proportion_outcomes <- c("score_recog_memory_imm", "score_recog_memory_del")

all_outcomes  <- c(continuous_outcomes, proportion_outcomes)
primary_vars  <- c(all_outcomes, "fibre_intake", "age", "sex", "energy_kcal")
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
#   all outcome variables listed in continuous_outcomes and proportion_outcomes

obs_data <- read_csv("data/derived/YOUR_FILE.csv") %>%
  mutate(id = as.factor(id))


# ---- Inspect data ------------------------------------------------------------

obs_data %>%
  select(all_of(all_outcomes)) %>%
  dfSummary() %>%
  stview()


# ---- Missing data pattern ----------------------------------------------------

obs_data %>%
  select(all_of(primary_vars)) %>%
  plot_pattern()

# ---- EDA: correlation matrix -------------------------------------------------

eda_cog <- obs_data %>%
  ggpairs(
    columns = c(
      "fibre_intake",
      "score_memory_total",    "score_memory_immediate", "score_memory_delayed",
      "score_wm_errors",       "score_wm_strategy",      "score_set_shifting",
      "score_attention",       "score_recog_memory_imm", "score_recog_memory_del",
      "score_emotion_recog",   "score_attentional_bias",
      "score_decision_making", "score_inhibition"
    ),
    columnLabels = c(
      "Fibre (g/day)",
      "Memory Total", "Memory Immediate", "Memory Delayed",
      "WM Errors",    "WM Strategy",      "Set Shifting",
      "Attention",    "Recog Imm (%)",    "Recog Del (%)",
      "Emotion Recog","Attentional Bias",
      "Decision Making", "Inhibition"
    ),
    lower = list(continuous = "smooth")
  )

ggsave("plots/eda_cognition.png", eda_cog, width = 14, height = 12, dpi = 300)


# ---- Descriptive table -------------------------------------------------------
# Outcomes are typically non-normal; Kruskal-Wallis used for unadjusted comparison between levels of fibre. This was done for descriptive purposes, no statistical tests were done for non-outcome variables in line with best practices for observational studies and CONSORT.

cog_desc_table <- obs_data %>%
  mutate(fibre_grouping = factor(fibre_grouping,
                                 levels = fibre_group_levels,
                                 labels = fibre_group_labels)) %>%
  select(fibre_grouping, all_of(all_outcomes)) %>%
  tbl_summary(
    by        = fibre_grouping,
    statistic = list(all_continuous() ~ "{median} ({p25}, {p75})"),
    label     = list(
      score_memory_total     ~ "Verbal Learning Total",
      score_memory_immediate ~ "Verbal Learning Immediate",
      score_memory_delayed   ~ "Verbal Learning Delayed Recall",
      score_wm_errors        ~ "Working Memory Errors",
      score_wm_strategy      ~ "Working Memory Strategy",
      score_set_shifting     ~ "Set Shifting Errors",
      score_attention        ~ "Sustained Attention",
      score_recog_memory_imm ~ "Recognition Memory Immediate (%)",
      score_recog_memory_del ~ "Recognition Memory Delayed (%)",
      score_emotion_recog    ~ "Emotion Recognition Hits",
      score_attentional_bias ~ "Attentional Bias Score",
      score_decision_making  ~ "Decision Making Performance",
      score_inhibition       ~ "Inhibition Effect"
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

cog_desc_table %>%
  as_flex_table() %>%
  save_as_docx(path = "tables/OBS_cognition_descriptives.docx")


# ---- Data preparation --------------------------------------------------------

# Complete-case dataset (minimum variables required for basic models)
obs_cc <- obs_data %>%
  drop_na(all_of(primary_vars))

# Mean-centre continuous predictors and covariates.
# scale = FALSE centres without dividing by SD, preserving original units so
# that model coefficients reflect change per unit of the original variable.
# 
obs_cc <- obs_cc %>%
  mutate(
    fibre_c           = scale(fibre_intake, center = TRUE, scale = FALSE),
    age_c             = scale(age,          center = TRUE, scale = FALSE),
    energy_c          = scale(energy_kcal,  center = TRUE, scale = FALSE),
    pal_c             = scale(pal,          center = TRUE, scale = FALSE),
    sex               = as.factor(sex),
    employment_status = factor(employment_status, levels = employment_levels),
    # Rescale proportion outcomes to 0–1 for beta-binomial models
    score_recog_memory_imm_prop = score_recog_memory_imm / 100,
    score_recog_memory_del_prop = score_recog_memory_del / 100,
    trials                      = prop_trials
  )

# Save model-ready dataset (dated for reproducibility)
write.csv(
  obs_cc,
  file      = paste0("data/derived/obs_cog_data_", Sys.Date(), ".csv"),
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
#       on the adjusted if there is any reason to suspect otherwise.)
#   3. Export basic and adjusted side-by-side with tab_model().
# =============================================================================


# ---- Memory: Total Score (Gaussian lm) ---------------------------------------

mod_memory_total <- lm(
  score_memory_total ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_memory_total_adj <- lm(
  score_memory_total ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_memory_total)
res_memory_total <- simulateResiduals(mod_memory_total, n = 1000)
testResiduals(res_memory_total)

tab_model(
  mod_memory_total, mod_memory_total_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_memory_total.doc"
)


# ---- Memory: Immediate Recall (Gaussian lm) ----------------------------------

mod_memory_immediate <- lm(
  score_memory_immediate ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_memory_immediate_adj <- lm(
  score_memory_immediate ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_memory_immediate)
res_memory_immediate <- simulateResiduals(mod_memory_immediate, n = 1000)
testResiduals(res_memory_immediate)

tab_model(
  mod_memory_immediate, mod_memory_immediate_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_memory_immediate.doc"
)


# ---- Memory: Delayed Recall (Gaussian lm) ------------------------------------

mod_memory_delayed <- lm(
  score_memory_delayed ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_memory_delayed_adj <- lm(
  score_memory_delayed ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_memory_delayed)
res_memory_delayed <- simulateResiduals(mod_memory_delayed, n = 1000)
testResiduals(res_memory_delayed)

tab_model(
  mod_memory_delayed, mod_memory_delayed_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_memory_delayed.doc"
)


# ---- Working Memory: Between Errors (Gaussian lm) ----------------------------
# Error counts may show right skew; check_distribution() is included to flag
# whether a count-family model (Poisson / NB) would be more appropriate.
# If residuals show strong non-normality, follow the ZINB workflow in the
# companion obs_mood_template.R.

mod_wm_errors <- lm(
  score_wm_errors ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_wm_errors_adj <- lm(
  score_wm_errors ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_wm_errors)
check_distribution(mod_wm_errors)
res_wm_errors <- simulateResiduals(mod_wm_errors, n = 1000)
testResiduals(res_wm_errors)

tab_model(
  mod_wm_errors, mod_wm_errors_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_wm_errors.doc"
)


# ---- Working Memory: Strategy Score (Gaussian lm) ----------------------------

mod_wm_strategy <- lm(
  score_wm_strategy ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_wm_strategy_adj <- lm(
  score_wm_strategy ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_wm_strategy)
res_wm_strategy <- simulateResiduals(mod_wm_strategy, n = 1000)
testResiduals(res_wm_strategy)

tab_model(
  mod_wm_strategy, mod_wm_strategy_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_wm_strategy.doc"
)


# ---- Set Shifting: Adjusted Errors (Gaussian lm, log-transformed outcome) ----
# The error count is positively skewed; a log transformation is applied directly
# to the outcome in the model formula to improve residual normality.
# Coefficients reflect effects on the log-scale error count. If residuals remain
# non-normal after log-transformation, consider a Poisson or negative binomial
# model via glmmTMB 

mod_set_shifting <- lm(
  log(score_set_shifting) ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_set_shifting_adj <- lm(
  log(score_set_shifting) ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_set_shifting)
res_set_shifting <- simulateResiduals(mod_set_shifting, n = 1000)
testResiduals(res_set_shifting)

tab_model(
  mod_set_shifting, mod_set_shifting_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_set_shifting.doc"
)


# ---- Sustained Attention: Target Detection Probability (Gaussian lm) --------

mod_attention <- lm(
  score_attention ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_attention_adj <- lm(
  score_attention ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_attention)
res_attention <- simulateResiduals(mod_attention, n = 1000)
testResiduals(res_attention)

tab_model(
  mod_attention, mod_attention_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_attention.doc"
)


# ---- Emotion Recognition: Total Hits (Gaussian lm) --------------------------

mod_emotion_recog <- lm(
  score_emotion_recog ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_emotion_recog_adj <- lm(
  score_emotion_recog ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_emotion_recog)
res_emotion_recog <- simulateResiduals(mod_emotion_recog, n = 1000)
testResiduals(res_emotion_recog)

tab_model(
  mod_emotion_recog, mod_emotion_recog_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_emotion_recog.doc"
)


# ---- Attentional Bias Score (Gaussian lm) ------------------------------------

mod_attentional_bias <- lm(
  score_attentional_bias ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_attentional_bias_adj <- lm(
  score_attentional_bias ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_attentional_bias)
res_attentional_bias <- simulateResiduals(mod_attentional_bias, n = 1000)
testResiduals(res_attentional_bias)

tab_model(
  mod_attentional_bias, mod_attentional_bias_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_attentional_bias.doc"
)


# ---- Decision Making: Total Performance (Gaussian lm) -----------------------

mod_decision_making <- lm(
  score_decision_making ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_decision_making_adj <- lm(
  score_decision_making ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_decision_making)
res_decision_making <- simulateResiduals(mod_decision_making, n = 1000)
testResiduals(res_decision_making)

tab_model(
  mod_decision_making, mod_decision_making_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_decision_making.doc"
)


# ---- Inhibitory Control: Interference Effect (Gaussian lm) ------------------

mod_inhibition <- lm(
  score_inhibition ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
mod_inhibition_adj <- lm(
  score_inhibition ~ fibre_c + sex + age_c + energy_c + employment_status + pal_c,
  data = obs_cc
)

check_model(mod_inhibition)
res_inhibition <- simulateResiduals(mod_inhibition, n = 1000)
testResiduals(res_inhibition)

tab_model(
  mod_inhibition, mod_inhibition_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_inhibition.doc"
)


# ---- Recognition Memory: Proportion Outcomes (beta-binomial / glmmTMB) ------
# score_recog_memory_imm and score_recog_memory_del are percentage-correct
# scores bounded between 0 and 100, arising from a fixed number of test trials.
# Gaussian lm is fitted first as a diagnostic baseline; residuals typically
# show boundary compression and non-normality, motivating a beta-binomial model.
#
# Beta-binomial regression via glmmTMB:
#   - The proportion (0–1) is modelled against the number of trials (weights).
#   - Beta-binomial accommodates overdispersion relative to pure binomial,
#     which is common in cognitive test accuracy data.
#   - Estimates are on the log-odds scale; tab_model() back-transforms to
#     odds ratios. Use type = "response" in emmeans for probability scale.
#   - check_overdispersion() and DHARMa confirm model fit.
#
# prop_trials and the _prop columns are defined in the data preparation section.


# --- Immediate recognition memory ---

# Step 1 — Gaussian baseline check
mod_recog_memory_imm_lm <- lm(
  score_recog_memory_imm ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
check_model(mod_recog_memory_imm_lm)
res_recog_imm_lm <- simulateResiduals(mod_recog_memory_imm_lm, n = 1000)
plot(res_recog_imm_lm)

# Step 2 — Beta-binomial
mod_recog_memory_imm <- glmmTMB(
  score_recog_memory_imm_prop ~ fibre_c + sex + age_c + energy_c,
  family  = betabinomial(link = "logit"),
  weights = trials,
  data    = obs_cc
)
mod_recog_memory_imm_adj <- glmmTMB(
  score_recog_memory_imm_prop ~ fibre_c + sex + age_c + energy_c +
    employment_status + pal_c,
  family  = betabinomial(link = "logit"),
  weights = trials,
  data    = obs_cc
)

check_overdispersion(mod_recog_memory_imm)
res_recog_imm <- simulateResiduals(mod_recog_memory_imm, n = 1000)
testResiduals(res_recog_imm)

tab_model(
  mod_recog_memory_imm, mod_recog_memory_imm_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_recog_memory_imm.doc"
)


# --- Delayed recognition memory ---

# Step 1 — Gaussian baseline check
mod_recog_memory_del_lm <- lm(
  score_recog_memory_del ~ fibre_c + sex + age_c + energy_c,
  data = obs_cc
)
check_model(mod_recog_memory_del_lm)
res_recog_del_lm <- simulateResiduals(mod_recog_memory_del_lm, n = 1000)
plot(res_recog_del_lm)

# Step 2 — Beta-binomial
mod_recog_memory_del <- glmmTMB(
  score_recog_memory_del_prop ~ fibre_c + sex + age_c + energy_c,
  family  = betabinomial(link = "logit"),
  weights = trials,
  data    = obs_cc
)
mod_recog_memory_del_adj <- glmmTMB(
  score_recog_memory_del_prop ~ fibre_c + sex + age_c + energy_c +
    employment_status + pal_c,
  family  = betabinomial(link = "logit"),
  weights = trials,
  data    = obs_cc
)

check_overdispersion(mod_recog_memory_del)
res_recog_del <- simulateResiduals(mod_recog_memory_del, n = 1000)
testResiduals(res_recog_del)

tab_model(
  mod_recog_memory_del, mod_recog_memory_del_adj,
  dv.labels   = c("Unadjusted", "Adjusted"),
  show.reflvl = TRUE,
  file        = "tables/OBS_recog_memory_del.doc"
)
