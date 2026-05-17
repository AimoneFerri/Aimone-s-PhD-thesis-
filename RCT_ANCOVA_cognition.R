# =============================================================================
# ANCOVA models for cognitive outcomes
# Generalized template for RCT
# =============================================================================
#
# Design: two timepoints (baseline → follow-up) pre-post comparison.
#Long-format data is reshaped to wide; ANCOVA is implemented as
#standard lm with the centred baseline score as a covariate.
#
# Model structure (one model per outcome):
#   outcome_fu ~ baseline_c + group + sex + age_c
#
# Outcomes modelled:
#   Continuous Gaussian (lm):
#     score_memory_total,    score_memory_immediate, score_memory_delayed,
#     score_wm_errors,       score_wm_strategy,      score_set_shifting,
#     score_attention,       score_emotion_recog,    score_attentional_bias,
#     score_decision_making, score_inhibition
#   Bounded proportion (0–100 %) — beta-binomial glmmTMB (see dedicated section):
#     score_recog_memory_imm, score_recog_memory_del
# =============================================================================


# ---- load libs ---------------------------------------------------------------

library(readr)
library(dplyr)
library(tidyr)
library(summarytools)
library(ggmice)
library(ggplot2)
library(viridis)
library(glmmTMB)
library(easystats)
library(sjPlot)
library(DHARMa)
library(emmeans) #if needed for post-hoc contrasts 

options(scipen = 999)


# ---- Study design constants --------------------------------------------------
# Adapt visit codes, group codes, and group labels to your dataset

baseline_visit <- 2
followup_visit <- 4

# Construct visit label lookup from constants
visit_recode <- setNames(
  c("bl", "fu"),
  as.character(c(baseline_visit, followup_visit))
)
timepoint_codes <- as.character(c(baseline_visit, followup_visit))

# Adapt group codes (left) and display labels (right) to your study
group_recode <- c(
  "ICTR"  = "Control",
  "IFIB"  = "High Fiber",
  "IFER"  = "Fermented",
  "ICOMB" = "Combined"
)
group_levels_ordered <- c("Control", "High Fiber", "Fermented", "Combined")

# Outcome variable names as they appear in the long-format input file.
# These become <outcome>_bl and <outcome>_fu after reshaping to wide format.

outcomes <- c(
  "score_memory_total",
  "score_memory_immediate",
  "score_memory_delayed",
  "score_wm_errors",
  "score_wm_strategy",
  "score_set_shifting",
  "score_attention",
  "score_recog_memory_imm",    # bounded proportion (0–100%) — see beta-binomial section
  "score_recog_memory_del",    # bounded proportion (0–100%) — see beta-binomial section
  "score_emotion_recog",
  "score_attentional_bias",
  "score_decision_making",
  "score_inhibition"
)

# ---- Load data ---------------------------------------------------------------
# Required columns: id, visit, group, sex, age, all outcomes above.
# Data should be in long format as per data files (one row per participant per visit).

nmb_data <- read_csv("data/derived/YOUR_FILE.csv") %>%
  mutate(
    id = as.factor(id),
    visit          = as.factor(visit)
  )


# ---- Inspect data ------------------------------------------------------------
# Quick descriptive summary of outcome variables at all timepoints
nmb_data %>%
  select(all_of(outcomes)) %>%
  dfSummary() %>%
  stview()

# ---- Missing data pattern ----------------------------------------------------

# Visualise missingness at baseline
nmb_data %>%
  filter(visit == as.character(baseline_visit)) %>%
  select(all_of(outcomes)) %>%
  plot_pattern()

# Count complete cases at baseline
nmb_data %>%
  filter(visit == as.character(baseline_visit)) %>%
  summarise(
    n_complete = sum(complete.cases(across(all_of(outcomes)))),
    n_total    = n()
  )

# Remove rows with missing visit code (incomplete)
nmb_data <- nmb_data %>%
  filter(!is.na(visit))


# ---- Trajectory plots --------------------------------------------------------
# Inspect pre-to-post trajectories per group before modelling.
# Filters to participants observed at both timepoints with no missing outcome.

# Generic pre-post spaghetti + group-mean plot function
trajectory_plot <- function(data, outcome_var, y_label) {
  data %>%
    filter(visit %in% timepoint_codes) %>%
    group_by(id) %>%
    filter(
      n() == length(timepoint_codes),
      !any(is.na(.data[[outcome_var]]))
    ) %>%
    ungroup() %>%
    mutate(
      group     = recode(group, !!!group_recode),
      group     = factor(group, levels = group_levels_ordered),
      timepoint = recode(visit, "bl" = "Baseline", "fu" = "Follow-up",
                         !!as.character(baseline_visit) := "Baseline",
                         !!as.character(followup_visit)  := "Follow-up"),
      timepoint = factor(timepoint, levels = c("Baseline", "Follow-up"))
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

# Generate and save one plot per outcome
plot_memory_total      <- trajectory_plot(nmb_data, "score_memory_total",      "Memory Total Score")
plot_memory_immediate  <- trajectory_plot(nmb_data, "score_memory_immediate",  "Memory Immediate Score")
plot_memory_delayed    <- trajectory_plot(nmb_data, "score_memory_delayed",    "Memory Delayed Score")
plot_wm_errors         <- trajectory_plot(nmb_data, "score_wm_errors",         "Working Memory Errors")
plot_wm_strategy       <- trajectory_plot(nmb_data, "score_wm_strategy",       "Working Memory Strategy")
plot_set_shifting      <- trajectory_plot(nmb_data, "score_set_shifting",      "Set-Shifting Errors")
plot_attention         <- trajectory_plot(nmb_data, "score_attention",         "Attention Score")
plot_recog_memory_imm  <- trajectory_plot(nmb_data, "score_recog_memory_imm",  "Recognition Memory Immediate (%)")
plot_recog_memory_del  <- trajectory_plot(nmb_data, "score_recog_memory_del",  "Recognition Memory Delayed (%)")
plot_emotion_recog     <- trajectory_plot(nmb_data, "score_emotion_recog",     "Emotion Recognition Score")
plot_attentional_bias  <- trajectory_plot(nmb_data, "score_attentional_bias",  "Attentional Bias Score")
plot_decision_making   <- trajectory_plot(nmb_data, "score_decision_making",   "Decision-Making Score")
plot_inhibition        <- trajectory_plot(nmb_data, "score_inhibition",        "Inhibition Effect (ms)")

ggsave("plots/plot_memory_total.png",      plot_memory_total,      width = 7, height = 6, dpi = 300)
ggsave("plots/plot_memory_immediate.png",  plot_memory_immediate,  width = 7, height = 6, dpi = 300)
ggsave("plots/plot_memory_delayed.png",    plot_memory_delayed,    width = 7, height = 6, dpi = 300)
ggsave("plots/plot_wm_errors.png",         plot_wm_errors,         width = 7, height = 6, dpi = 300)
ggsave("plots/plot_wm_strategy.png",       plot_wm_strategy,       width = 7, height = 6, dpi = 300)
ggsave("plots/plot_set_shifting.png",      plot_set_shifting,      width = 7, height = 6, dpi = 300)
ggsave("plots/plot_attention.png",         plot_attention,         width = 7, height = 6, dpi = 300)
ggsave("plots/plot_recog_memory_imm.png",  plot_recog_memory_imm,  width = 7, height = 6, dpi = 300)
ggsave("plots/plot_recog_memory_del.png",  plot_recog_memory_del,  width = 7, height = 6, dpi = 300)
ggsave("plots/plot_emotion_recog.png",     plot_emotion_recog,     width = 7, height = 6, dpi = 300)
ggsave("plots/plot_attentional_bias.png",  plot_attentional_bias,  width = 7, height = 6, dpi = 300)
ggsave("plots/plot_decision_making.png",   plot_decision_making,   width = 7, height = 6, dpi = 300)
ggsave("plots/plot_inhibition.png",        plot_inhibition,        width = 7, height = 6, dpi = 300)


# ---- Reshape to wide format --------------------------------------------------
# ANCOVA requires one row per participant with baseline and follow-up as
# separate columns. Visits are relabelled to "bl" / "fu" before pivoting

ancova_data <- nmb_data %>%
  filter(visit %in% timepoint_codes) %>%
  mutate(visit = recode(visit, !!!visit_recode)) %>%
  select(id, visit, group, sex, age, all_of(outcomes)) %>%
  pivot_wider(
    id_cols     = c(id, group, sex, age),
    names_from  = visit,
    values_from = all_of(outcomes)
  )


# ---- Centre baseline covariates and age --------------------------------------
# scale(x, scale = FALSE) centres without dividing by SD
# The centred baseline covariate of the response variable improves precision of the estimate and adjusts for possible natural imbalances between groups. Centring age makes coefficient interpretable.

ancova_data <- ancova_data %>%
  mutate(
    score_memory_total_bl_c      = scale(score_memory_total_bl,      scale = FALSE),
    score_memory_immediate_bl_c  = scale(score_memory_immediate_bl,  scale = FALSE),
    score_memory_delayed_bl_c    = scale(score_memory_delayed_bl,    scale = FALSE),
    score_wm_errors_bl_c         = scale(score_wm_errors_bl,         scale = FALSE),
    score_wm_strategy_bl_c       = scale(score_wm_strategy_bl,       scale = FALSE),
    score_set_shifting_bl_c      = scale(score_set_shifting_bl,      scale = FALSE),
    score_attention_bl_c         = scale(score_attention_bl,         scale = FALSE),
    score_recog_memory_imm_bl_c  = scale(score_recog_memory_imm_bl,  scale = FALSE),
    score_recog_memory_del_bl_c  = scale(score_recog_memory_del_bl,  scale = FALSE),
    score_emotion_recog_bl_c     = scale(score_emotion_recog_bl,     scale = FALSE),
    score_attentional_bias_bl_c  = scale(score_attentional_bias_bl,  scale = FALSE),
    score_decision_making_bl_c   = scale(score_decision_making_bl,   scale = FALSE),
    score_inhibition_bl_c        = scale(score_inhibition_bl,        scale = FALSE),
    age_c                        = scale(age,                        scale = FALSE)
  ) %>%
  mutate(
    group = relevel(factor(group), ref = "ICTR"),   # ICTR = control arm
    sex   = as.factor(sex)
  )


# ---- Save model-ready dataset (dated) ----------------------------------------

write.csv(
  ancova_data,
  file = paste0("data/derived/ancova_data_", Sys.Date(), ".csv"),
  row.names = FALSE
)

# =============================================================================
# ANCOVA MODELS
# =============================================================================
# Structure: outcome_fu ~ baseline_c + group + sex + age_c
#   - outcome_fu  : follow-up score (post-intervention)
#   - baseline_c  : mean-centred baseline score (covariate)
#   - group       : intervention allocation (reference = control)
#   - sex, age_c  : additional covariates
#
# lm() is used for all continuous outcomes. The lm ANCOVA is appropriate
# when there are exactly two timepoints (baseline + follow-up)
# 
# Diagnostics:
#   check_model()          — visual overview (normality, linearity, leverage)
#   simulateResiduals()    — simulation-based quantile residuals (DHARMa)
#   testResiduals()        — formal tests for uniformity, dispersion, outliers
#   testZeroInflation()    — applied where excess zeros are plausible
# =============================================================================


# ---- Memory: Total Score (Gaussian lm) ---------------------------------------

mod_memory_total <- lm(
  score_memory_total_fu ~ score_memory_total_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_memory_total)
check_model(mod_memory_total)
res_memory_total <- simulateResiduals(mod_memory_total)
plot(res_memory_total)
testResiduals(res_memory_total)


# ---- Memory: Immediate Recall (Gaussian lm) ----------------------------------

mod_memory_immediate <- lm(
  score_memory_immediate_fu ~ score_memory_immediate_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_memory_immediate)
check_model(mod_memory_immediate)
res_memory_immediate <- simulateResiduals(mod_memory_immediate)
plot(res_memory_immediate)
testResiduals(res_memory_immediate)


# ---- Memory: Delayed Recall (Gaussian lm) ------------------------------------

mod_memory_delayed <- lm(
  score_memory_delayed_fu ~ score_memory_delayed_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_memory_delayed)
check_model(mod_memory_delayed)
res_memory_delayed <- simulateResiduals(mod_memory_delayed)
plot(res_memory_delayed)
testResiduals(res_memory_delayed)


# ---- Working Memory: Between Errors (Gaussian lm) ----------------------------
# Error counts may show right skew or excess zeros; check both below.

mod_wm_errors <- lm(
  score_wm_errors_fu ~ score_wm_errors_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_wm_errors)
check_model(mod_wm_errors)
res_wm_errors <- simulateResiduals(mod_wm_errors)
plot(res_wm_errors)
testResiduals(res_wm_errors)
testZeroInflation(res_wm_errors)


# ---- Working Memory: Strategy Score (Gaussian lm) ----------------------------

mod_wm_strategy <- lm(
  score_wm_strategy_fu ~ score_wm_strategy_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_wm_strategy)
check_model(mod_wm_strategy)
res_wm_strategy <- simulateResiduals(mod_wm_strategy)
plot(res_wm_strategy)
testResiduals(res_wm_strategy)


# ---- Set-Shifting: Adjusted Errors (Gaussian lm) -----------------------------
# This outcome is typically positively skewed (right-tailed count of errors).
# check_distribution() is included to flag whether a count-family model
# (Poisson / negative binomial via glmmTMB) would be more appropriate.
# A simple log transform often works best with this outcome. 

mod_set_shifting <- lm(
  score_set_shifting_fu ~ score_set_shifting_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_set_shifting)
check_model(mod_set_shifting)
check_distribution(mod_set_shifting)
res_set_shifting <- simulateResiduals(mod_set_shifting)
plot(res_set_shifting)
testResiduals(res_set_shifting)


# ---- Attention: Target Detection Probability (Gaussian lm) -------------------

mod_attention <- lm(
  score_attention_fu ~ score_attention_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_attention)
check_model(mod_attention)
res_attention <- simulateResiduals(mod_attention)
plot(res_attention)
testResiduals(res_attention)


# ---- Emotion Recognition: Total Hits (Gaussian lm) --------------------------

mod_emotion_recog <- lm(
  score_emotion_recog_fu ~ score_emotion_recog_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_emotion_recog)
check_model(mod_emotion_recog)
res_emotion_recog <- simulateResiduals(mod_emotion_recog)
plot(res_emotion_recog)
testResiduals(res_emotion_recog)


# ---- Attentional Bias Score (Gaussian lm) ------------------------------------

mod_attentional_bias <- lm(
  score_attentional_bias_fu ~ score_attentional_bias_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_attentional_bias)
check_model(mod_attentional_bias)
res_attentional_bias <- simulateResiduals(mod_attentional_bias)
plot(res_attentional_bias)
testResiduals(res_attentional_bias)


# ---- Decision-Making: Total Performance (Gaussian lm) ------------------------

mod_decision_making <- lm(
  score_decision_making_fu ~ score_decision_making_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_decision_making)
check_model(mod_decision_making)
res_decision_making <- simulateResiduals(mod_decision_making)
plot(res_decision_making)
testResiduals(res_decision_making)


# ---- Inhibition: Interference Effect (Gaussian lm) --------------------------

mod_inhibition <- lm(
  score_inhibition_fu ~ score_inhibition_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_inhibition)
check_model(mod_inhibition)
res_inhibition <- simulateResiduals(mod_inhibition)
plot(res_inhibition)
testResiduals(res_inhibition)


# ---- Recognition Memory: Proportion Outcomes (beta-binomial / glmmTMB) -------
# score_recog_memory_imm and score_recog_memory_del are percentage-correct
# scores bounded between 0 and 100 (proportion data from a fixed number of
# trials). Gaussian lm is fitted first as a diagnostic baseline; residual
# diagnostics typically reveal boundary compression and non-normality
# Recommended model: beta-binomial regression via glmmTMB.
#   - Rescale the percentage score to a 0–1 proportion (divide by 100).
#   - Supply the number of test trials as a weights argument.
#   - Beta-binomial accommodates overdispersion relative to pure binomial,
#     which is common in cognitive test accuracy data.
#   - Estimates are on the log-odds scale; tab_model() back-transforms to
#     odds ratios. Use type = "response" in emmeans for the probability scale.
#   - check_overdispersion() and simulateResiduals() confirm model fit.
#

prop_trials <- 18   # number of test trials per participant — adapt as needed

ancova_data <- ancova_data %>%
  mutate(
    score_recog_memory_imm_prop = score_recog_memory_imm_fu / 100,
    score_recog_memory_del_prop = score_recog_memory_del_fu / 100,
    trials                      = prop_trials
  )


# --- Immediate recognition memory ---

# Step 1 — Gaussian baseline check, it will look poorly fitted
mod_recog_memory_imm_lm <- lm(
  score_recog_memory_imm_fu ~ score_recog_memory_imm_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_recog_memory_imm_lm)
check_model(mod_recog_memory_imm_lm)
res_recog_imm_lm <- simulateResiduals(mod_recog_memory_imm_lm)
plot(res_recog_imm_lm)

# Step 2 — Beta-binomial
mod_recog_memory_imm <- glmmTMB(
  score_recog_memory_imm_prop ~ score_recog_memory_imm_bl_c + group + sex + age_c,
  family  = betabinomial(link = "logit"),
  weights = trials,
  data    = ancova_data
)
tab_model(mod_recog_memory_imm)
check_model(mod_recog_memory_imm)
check_overdispersion(mod_recog_memory_imm)
res_recog_imm <- simulateResiduals(mod_recog_memory_imm)
plot(res_recog_imm)
testResiduals(res_recog_imm)


# --- Delayed recognition memory ---

# Step 1 — Gaussian baseline check
mod_recog_memory_del_lm <- lm(
  score_recog_memory_del_fu ~ score_recog_memory_del_bl_c + group + sex + age_c,
  data = ancova_data
)
tab_model(mod_recog_memory_del_lm)
check_model(mod_recog_memory_del_lm)
res_recog_del_lm <- simulateResiduals(mod_recog_memory_del_lm)
plot(res_recog_del_lm)

# Step 2 — Beta-binomial
mod_recog_memory_del <- glmmTMB(
  score_recog_memory_del_prop ~ score_recog_memory_del_bl_c + group + sex + age_c,
  family  = betabinomial(link = "logit"),
  weights = trials,
  data    = ancova_data
)
tab_model(mod_recog_memory_del)
check_model(mod_recog_memory_del)
check_overdispersion(mod_recog_memory_del)
res_recog_del <- simulateResiduals(mod_recog_memory_del)
plot(res_recog_del)
testResiduals(res_recog_del)


# ---- Combined model output tables --------------------------------------------
# Split across two calls to keep tables readable.
# Use the beta-binomial models (not lm) for recognition memory outcomes.

tab_model(
  mod_memory_total, mod_memory_immediate, mod_memory_delayed,
  mod_attention,    mod_wm_errors,        mod_wm_strategy,
  mod_recog_memory_imm, mod_recog_memory_del,
  dv.labels = c("Memory Total", "Memory Immediate", "Memory Delayed",
                "Attention", "WM Errors", "WM Strategy",
                "Recog Memory Imm", "Recog Memory Del")
)

tab_model(
  mod_set_shifting,     mod_emotion_recog,
  mod_attentional_bias, mod_decision_making, mod_inhibition,
  dv.labels = c("Set Shifting", "Emotion Recognition",
                "Attentional Bias", "Decision Making", "Inhibition")
)


# ---- Estimated marginal means and contrasts ----------------------------------
# ANCOVA has no timepoint dimension; emmeans marginalises over covariates only.
# Contrasts compare each active group to the control arm at the mean of all
# continuous covariates. type = "response" back-transforms beta-binomial
# estimates to the probability scale; lm estimates remain on the raw scale.

# Quick visual check of marginal means for a single model
emmip(mod_memory_total, ~ group)
emmeans(mod_memory_total, ~ group)

# Generic contrast extraction for a single ANCOVA model.
# Detects estimate, CI, and test-statistic column names dynamically so it
# works with both lm (difference scale) and glmmTMB (ratio / odds-ratio scale).
get_contrasts <- function(model, outcome_name, ref_level = "ICTR") {

  emm  <- emmeans(model, ~ group, type = "response")
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
      Outcome  = outcome,
      Contrast = contrast,
      Estimate = !!est_col,
      SE       = SE,
      DF       = !!df_col,
      Stat     = !!stat_col,
      p_value  = p.value,
      CI_lower = !!ci_lower_col,
      CI_upper = !!ci_upper_col
    )
}

# Example: extract and print contrasts for the memory total model
contrasts_memory_total <- get_contrasts(mod_memory_total, "Memory Total")
print(contrasts_memory_total)


# ---- EMM contrast plot (single model) ----------------------------------------
# Plots each active group's contrast vs. control with 95% CIs.
# Run separately per outcome; adapt model object, title, and ref level.

plot(
  contrast(
    emmeans(mod_memory_total, ~ group),
    method = "trt.vs.ctrl",
    ref    = "CTR",
    adjust = "none"
  ),
  comparisons = TRUE,
  CIs         = TRUE,
  type        = "response"
) +
  labs(title = "Memory Total: group contrasts vs. control") +
  theme(plot.title = element_text(size = 12, face = "bold"))

ggsave("plots/contrast_memory_total.png", width = 7, height = 5, dpi = 300)
