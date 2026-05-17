# =============================================================================
# Mixed-Effects Models: Psychological / Mood Outcomes
# Generic Template — Repeated-Measures Intervention RCT
# =============================================================================
#
# Outcomes modelled (rename to match your variables):
#   score_stress      : continuous outcome → Gaussian lmer
#   score_anxiety     : continuous outcome → Gaussian lmer
#   score_symptoms    : right-skewed continuous → lognormal glmmTMB
#   score_depression  : technically count outcome → Gaussian checked first,
#                       then Poisson or negative binomial (see section below)
#
# General model structure (additive first, then interaction if motivated or graphically evident):
#   outcome ~ baseline_c + group + timepoint + sex + age_c + (1 | id)
#   outcome ~ baseline_c + group * timepoint + sex + age_c + (1 | id)
#
# Random intercept per participant accounts for within-person correlation (with 2 actual repeated measures any structure fits the same, so no need to specify structure with nlme)
# Centred baseline score improves precision of estimates and possible baseline imbalances between groups.
# Inspect trajectory plots before deciding whether to include the interaction.
# =============================================================================


# ---- Libraries ---------------------------------------------------------------

library(readr)
library(dplyr)
library(summarytools)
library(ggmice)
library(ggplot2)
library(viridis)
library(lme4)
library(glmmTMB)
library(DHARMa)
library(easystats)
library(sjPlot)
library(emmeans)

options(scipen = 999)


# ---- Load data ---------------------------------------------------------------

# Adapt file path and variable names to your dataset.
# Required columns: id, timepoint, group, sex, age,
#   score_stress, score_anxiety, score_symptoms, score_depression

nmb_data <- read_csv("data/derived/YOUR_FILE.csv") %>%
  mutate(
    id = as.factor(id),
    timepoint      = as.factor(timepoint)
  )


# ---- Inspect data ------------------------------------------------------------

# Quick descriptive summary of outcome variables
nmb_data %>%
  select(score_depression, score_stress, score_anxiety, score_symptoms) %>%
  dfSummary() %>%
  stview()


# ---- Missing data ------------------------------------------------------------

# Visualise missing data pattern at baseline (adapt timepoint label as needed)
nmb_data %>%
  filter(timepoint == "2") %>%
  select(score_depression, score_stress, score_anxiety, score_symptoms) %>%
  plot_pattern()

# Count complete cases at baseline
nmb_data %>%
  filter(timepoint == "2") %>%
  summarise(
    n_complete = sum(complete.cases(score_depression, score_stress,
                                    score_anxiety,    score_symptoms)),
    n_total    = n()
  )

# ---- Trajectory plots --------------------------------------------------------
# Plot individual-level trajectories and group means for each outcome.
# Inspect these plots before fitting models:
#   - If group trajectories clearly diverge across timepoints, a
#     group × timepoint interaction is likely warranted.
#   - If trajectories run roughly parallel, the additive model may be
#     sufficient
#     Not considered here for practical and sample size reasons: possible non-linear effects may be modelled with baseline x group interaction and or polynomials/splines/generalized additive models with smoothing terms.

# Adapt group codes (left) and display labels (right) to your study
group_recode <- c(
  "ICTR"  = "Control",
  "IFIB"  = "High Fiber",
  "IFER"  = "Fermented",
  "ICOMB" = "Combined"
)
group_levels_ordered <- c("Control", "High Fiber", "Fermented", "Combined")

# Adapt timepoint codes (names) and display labels (values) to your study
timepoint_labels <- c("2" = "Baseline", "3" = "Week 4", "4" = "Week 8")
timepoint_codes  <- names(timepoint_labels)

# Generic spaghetti + group-mean trajectory plot function.
# Filters to participants observed at all timepoints with no missing outcome.
# 
trajectory_plot <- function(data, outcome_var, y_label) {
  data %>%
    filter(timepoint %in% timepoint_codes) %>%
    group_by(id) %>%
    filter(
      n() == length(timepoint_codes),
      !any(is.na(.data[[outcome_var]]))
    ) %>%
    ungroup() %>%
    mutate(
      group     = recode(group, !!!group_recode),
      group     = factor(group, levels = group_levels_ordered),
      timepoint = recode(timepoint, !!!timepoint_labels),
      timepoint = factor(timepoint, levels = unname(timepoint_labels))
    ) %>%
    ggplot(aes(x = timepoint, y = .data[[outcome_var]])) +
    geom_line(aes(group = id), colour = "grey80", alpha = 0.6) +
    geom_point(aes(colour = group), size = 1, alpha = 0.6, shape = 16) +
    stat_summary(aes(group = group, colour = group),
                 fun = mean, geom = "line", linewidth = 1.2) +
    facet_wrap(~group) +
    scale_colour_viridis_d(begin = 0.1, end = 0.9) +
    theme_minimal() +
    labs(y = y_label, x = "Timepoint", colour = "Group") +
    theme(
      legend.position = "none",
      strip.text      = element_text(size = 12, face = "bold"),
      axis.title      = element_text(size = 12, face = "bold"),
      axis.text       = element_text(size = 12),
      legend.title    = element_text(size = 12, face = "bold"),
      legend.text     = element_text(size = 12)
    )
}

plot_depression <- trajectory_plot(nmb_data, "score_depression", "Depression Score")
plot_stress     <- trajectory_plot(nmb_data, "score_stress",     "Stress Score")
plot_anxiety    <- trajectory_plot(nmb_data, "score_anxiety",    "Anxiety Score")
plot_symptoms   <- trajectory_plot(nmb_data, "score_symptoms",   "Symptom Score")

ggsave("plots/plot_depression.png", plot_depression, width = 7, height = 6, dpi = 300)
ggsave("plots/plot_stress.png",     plot_stress,     width = 7, height = 6, dpi = 300)
ggsave("plots/plot_anxiety.png",    plot_anxiety,    width = 7, height = 6, dpi = 300)
ggsave("plots/plot_symptoms.png",   plot_symptoms,   width = 7, height = 6, dpi = 300)


# ---- Data preparation for models ---------------------------------------------

# Broadcast each participant's baseline value (timepoint == "2") to all rows.
# This creates the covariate for the mixed model.
# Adapt "2" to whichever timepoint label represents your baseline.
nmb_data <- nmb_data %>%
  group_by(id) %>%
  mutate(
    baseline_depression = score_depression[timepoint == "2"],
    baseline_stress     = score_stress[timepoint == "2"],
    baseline_anxiety    = score_anxiety[timepoint == "2"],
    baseline_symptoms   = score_symptoms[timepoint == "2"]
  ) %>%
  ungroup()

# Mean-centre baseline covariates and age.
# scale = FALSE centres without dividing by SD

nmb_data <- nmb_data %>%
  mutate(
    baseline_depression_c = scale(baseline_depression, scale = FALSE),
    baseline_stress_c     = scale(baseline_stress,     scale = FALSE),
    baseline_anxiety_c    = scale(baseline_anxiety,    scale = FALSE),
    baseline_symptoms_c   = scale(baseline_symptoms,   scale = FALSE),
    age_c                 = scale(age,                 scale = FALSE)
  )

# Exclude baseline timepoint: it is now encoded as a covariate, not a repeated row. Needed to actually to account for baseline and improve precision, reduce RTM effects and wrong interpretation.
# Set reference level for group to the control arm (adapt ref = to your code).
# 
model_data <- nmb_data %>%
  filter(timepoint != "2") %>%
  mutate(
    group     = relevel(factor(group), ref = "ICTR"),   # ICTR = control arm
    sex       = as.factor(sex),
    timepoint = as.factor(timepoint)
  )


# ---- Save model-ready dataset (dated) ----------------------------------------

write.csv(
  model_data,
  file = paste0("data/derived/model_data_", Sys.Date(), ".csv"),
  row.names = FALSE
)


# =============================================================================
# MIXED MODELS
# =============================================================================
# Structure: outcome ~ baseline_c + group [* timepoint] + sex + age_c + (1 | id)
#
# For each outcome:
#   1. Additive model  — group and timepoint as main effects only
#   2. Interaction model — group × timepoint, allows trajectories to diverge
#
# Use trajectory plots above, BIC-AIC / likelihood-ratio test, and theoretical
# justification to decide which model to report.
# =============================================================================


# ---- Score: Stress (Gaussian / lmer) ----------------------------------------

# Additive model: group and timepoint effects assumed to be independent
mod_stress_add <- lmer(
  score_stress ~ baseline_stress_c + group + as.numeric(timepoint) + sex + age_c + (1 | id),
  REML = TRUE,
  data = model_data
)
tab_model(mod_stress_add)
check_model(mod_stress_add)
res_stress_add <- simulateResiduals(mod_stress_add)
testResiduals(res_stress_add)

# Interaction model: allows group differences to vary across timepoints
mod_stress <- lmer(
  score_stress ~ baseline_stress_c + group * timepoint + sex + age_c + (1 | id),
  REML = TRUE,
  data = model_data
)
tab_model(mod_stress)
check_model(mod_stress)
res_stress <- simulateResiduals(mod_stress)
testResiduals(res_stress)


# ---- Score: Anxiety (Gaussian / lmer) ----------------------------------------

# Additive model
mod_anxiety_add <- lmer(
  score_anxiety ~ baseline_anxiety_c + group + as.numeric(timepoint) + sex + age_c + (1 | id),
  REML = TRUE,
  data = model_data
)
tab_model(mod_anxiety_add)
check_model(mod_anxiety_add)
res_anxiety_add <- simulateResiduals(mod_anxiety_add)
testResiduals(res_anxiety_add)

# Interaction model
mod_anxiety <- lmer(
  score_anxiety ~ baseline_anxiety_c + group * timepoint + sex + age_c + (1 | id),
  REML = TRUE,
  data = model_data
)
tab_model(mod_anxiety)
check_model(mod_anxiety)
res_anxiety <- simulateResiduals(mod_anxiety)
testResiduals(res_anxiety)


# ---- Score: Symptoms (lognormal / glmmTMB) -----------------------------------
# Used for right-skewed continuous outcomes where a log link is appropriate.
# Gaussian lmer will typically show right-skewed residuals for such outcomes.
# Gamma(link = "log") and gaussian(link = "log") are alternatives worth comparing.

# Additive model
mod_symptoms_add <- glmmTMB(
  score_symptoms ~ baseline_symptoms_c + group + as.numeric(timepoint) + sex + age_c + (1 | id),
  family = lognormal(link = "log"),
  data   = model_data
)
tab_model(mod_symptoms_add)
check_model(mod_symptoms_add)
res_symptoms_add <- simulateResiduals(mod_symptoms_add)
testResiduals(res_symptoms_add)
testOverdispersion(res_symptoms_add)

# Interaction model
mod_symptoms <- glmmTMB(
  score_symptoms ~ baseline_symptoms_c + group * timepoint + sex + age_c + (1 | id),
  family = lognormal(link = "log"),
  data   = model_data
)
tab_model(mod_symptoms)
check_model(mod_symptoms)
res_symptoms <- simulateResiduals(mod_symptoms)
testResiduals(res_symptoms)
testOverdispersion(res_symptoms)


# ---- Score: Depression (count outcome / glmmTMB) ----------------------------
# Depression-type instruments (e.g., BDI-II ) produce non-negative
# integer scores with right skew and frequent zero values in the general or healthy population.
#
#   Step 1 — Fit Gaussian lmer as a diagnostic baseline.
#             Residual plots typically reveal non-normality, right skew,
#             and/or predicted values below zero, motivating a count model.
#   Step 2 — Fit Poisson via glmmTMB. If testOverdispersion() is significant,
#             move to negative binomial.
#   Step 3 — Fit negative binomial (nbinom1: linear overdispersion;
#             nbinom2: quadratic). Use compare_performance() to select.
#   Zero-inflation: always inspect testZeroInflation() and the DHARMa
#             Q-Q plot visually. If zero-inflation is confirmed, add
#             ziformula = ~1 allows the most basic modelling of zero-inflation in this case.
#   Final family selection should be based on DHARMa diagnostics,
#   AIC/BIC from compare_performance(), and convergence (diagnose()).

# Step 1 — Gaussian baseline (additive then interaction)
mod_depression_lm_add <- lmer(
  score_depression ~ baseline_depression_c + group + as.numeric(timepoint) + sex + age_c + (1 | id),
  REML = TRUE,
  data = model_data
)
tab_model(mod_depression_lm_add)
check_model(mod_depression_lm_add)
res_dep_lm_add <- simulateResiduals(mod_depression_lm_add)
plot(res_dep_lm_add)
testZeroInflation(res_dep_lm_add)

mod_depression_lm <- lmer(
  score_depression ~ baseline_depression_c + group * timepoint + sex + age_c + (1 | id),
  REML = TRUE,
  data = model_data
)
tab_model(mod_depression_lm)
check_model(mod_depression_lm)
res_dep_lm <- simulateResiduals(mod_depression_lm)
plot(res_dep_lm)
testZeroInflation(res_dep_lm)

# Step 2 — Poisson (additive then interaction)
mod_depression_p_add <- glmmTMB(
  score_depression ~ baseline_depression_c + group + as.numeric(timepoint) + sex + age_c + (1 | id),
  # ziformula = ~1,   # uncomment if zero-inflation is confirmed
  family = poisson,
  data   = model_data
)
res_dep_p_add <- simulateResiduals(mod_depression_p_add)
testOverdispersion(res_dep_p_add)
testZeroInflation(res_dep_p_add)

mod_depression_p <- glmmTMB(
  score_depression ~ baseline_depression_c + group * timepoint + sex + age_c + (1 | id),
  # ziformula = ~1,   # uncomment if zero-inflation is confirmed
  family = poisson,
  data   = model_data
)
res_dep_p <- simulateResiduals(mod_depression_p)
testOverdispersion(res_dep_p)
testZeroInflation(res_dep_p)

# Step 3 — Negative binomial, nbinom1 (additive then interaction)
# Try nbinom2 as an alternative if nbinom1 fit is poor
mod_depression_nb_add <- glmmTMB(
  score_depression ~ baseline_depression_c + group + as.numeric(timepoint) + sex + age_c + (1 | id),
  # ziformula = ~1,   # uncomment if zero-inflation is confirmed
  family = nbinom1,
  data   = model_data
)
check_overdispersion(mod_depression_nb_add)
diagnose(mod_depression_nb_add)
res_dep_nb_add <- simulateResiduals(mod_depression_nb_add)
plot(res_dep_nb_add)
testDispersion(res_dep_nb_add)
testZeroInflation(res_dep_nb_add)

mod_depression_nb <- glmmTMB(
  score_depression ~ baseline_depression_c + group * timepoint + sex + age_c + (1 | id),
  # ziformula = ~1,   # uncomment if zero-inflation is confirmed
  family = nbinom1,
  data   = model_data
)
diagnose(mod_depression_nb)
res_dep_nb <- simulateResiduals(mod_depression_nb)
plot(res_dep_nb)
testDispersion(res_dep_nb)
testZeroInflation(res_dep_nb)

# Compare Poisson vs. negative binomial (interaction models)
compare_performance(mod_depression_p, mod_depression_nb)

# Replace mod_depression_p below with whichever model passes diagnostics
tab_model(mod_depression_p)


# ---- Combined model output table ---------------------------------------------
# Collects all four preferred final (interaction) models into one formatted table.
# Replace individual model objects if a different family was selected above.

tab_model(
  mod_stress, mod_anxiety, mod_symptoms, mod_depression_p,
  dv.labels = c("Stress", "Anxiety", "Symptoms", "Depression")
)


# ---- Estimated marginal means and contrasts ----------------------------------
# Contrasts: each active group vs. control at each timepoint.
# type = "response" back-transforms estimates to original scale for non-Gaussian
# models (e.g., rate ratios for Poisson, multiplicative factors for lognormal).

# Relabel timepoint levels for readable emmeans output (adapt as needed)
model_data <- model_data %>%
  mutate(timepoint = factor(timepoint,
                            levels = c("3", "4"),
                            labels = c("Week 4", "Week 8")))

# Quick visual check of marginal means — useful before extracting contrasts
emmip(mod_stress, ~ group | timepoint)
emmeans(mod_stress, ~ group | timepoint)

# Generic contrast extraction function for a single model.
# Works with both Gaussian (difference scale) and count / log-link (ratio scale).
# Column names for CI bounds, test statistics, and estimate type are detected
# dynamically because they vary between lmer and glmmTMB model families.
get_contrasts <- function(model, outcome_name, ref_level = "ICTR") {

  emm  <- emmeans(model, ~ group | timepoint, type = "response")
  cont <- contrast(emm, method = "trt.vs.ctrl", ref = ref_level)

  cont_df <- summary(cont, type = "response", infer = TRUE,
                     level = 0.95, adjust = "none") %>%
    as.data.frame()

  # Detect CI column names (vary by model family and emmeans version)
  ci_lower_col <- intersect(c("lower.CL", "conf.low",  "asymp.LCL"), colnames(cont_df))[1]
  ci_upper_col <- intersect(c("upper.CL", "conf.high", "asymp.UCL"), colnames(cont_df))[1]

  # Detect test statistic and estimate columns
  stat_col <- intersect(c("t.ratio", "z.ratio"),  colnames(cont_df))[1]
  est_col  <- intersect(c("estimate", "ratio"),    colnames(cont_df))[1]

  # Detect df column (absent for z-distributed models)
  df_col <- intersect("df", colnames(cont_df))
  if (length(df_col) == 0) df_col <- NA_character_

  cont_df %>%
    mutate(outcome = outcome_name) %>%
    select(
      Outcome   = outcome,
      Timepoint = timepoint,
      Contrast  = contrast,
      Estimate  = !!est_col,
      SE        = SE,
      DF        = !!df_col,
      Stat      = !!stat_col,
      p_value   = p.value,
      CI_lower  = !!ci_lower_col,
      CI_upper  = !!ci_upper_col
    )
}

# Example: extract and print contrasts for the stress model
contrasts_stress <- get_contrasts(mod_stress, "Stress")
print(contrasts_stress)


# ---- EMM contrast plot (single model) ----------------------------------------
# Plots each active group's contrast vs. control with 95% CIs at each timepoint.
# Run separately per outcome; adapt model object, title, and ref level.

plot(
  contrast(
    emmeans(mod_stress, ~ group | timepoint),
    method = "trt.vs.ctrl",
    ref    = "CTR",
    adjust = "none"
  ),
  comparisons = TRUE,
  CIs         = TRUE,
  type        = "response"
) +
  labs(title = "Stress: group contrasts vs. control by timepoint") +
  theme(plot.title = element_text(size = 12, face = "bold"))

ggsave("plots/contrast_stress.png", width = 7, height = 5, dpi = 300)
