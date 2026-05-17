# =============================================================================
# Bayesian Multivariate Mixed Model: Mood / Psychological Outcomes
# Generic Template — Repeated-Measures Intervention RCT
# =============================================================================
#
# Fits a joint multivariate Bayesian model (brms / Stan) for three outcomes
# simultaneously, estimating their residual correlations. A preliminary
# in addition to theorethical sense, a correlation check at baseline should inform whether a joint model is
# warranted vs. modelling outcomes independently.
#
# Outcomes (rename to match your variables):
#   score_depression, score_stress, score_symptoms
#   All three are z-scored before modelling (for stability and interepreation).
#
# Model structure:
#   mvbind(dep_z, str_z, sym_z) ~
#     group + visit + age + sex +
#     bl_depression_z + bl_stress_z + bl_symptoms_z + (1 | id)
#
#   - group    : intervention allocation (reference = ICTR)
#   - visit    : coded numeric — linear time effect; use as.factor() if
#                non-linear or non-monotonic trends are expected
#   - bl_*_z   : participant's own baseline z-score per outcome (covariate)
#   - (1 | id) : random intercept per participant
#   - rescor   : residual correlations between outcomes estimated (LKJ prior)
# =============================================================================


# ---- Libraries ---------------------------------------------------------------

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(sjPlot)
library(brms)
library(cmdstanr)
library(tidybayes)
library(bayesplot)
library(bayestestR)
library(ggdist)
library(emmeans)
library(patchwork)

options(scipen = 999)


# ---- Load data ---------------------------------------------------------------
# Long-format dataset with all visits.
# Required columns: id, visit, group, sex, age,
#   score_depression, score_stress, score_symptoms

nmb_data <- read_csv("data/derived/YOUR_FILE.csv") %>%
  mutate(
    id    = as.factor(id),
    group = relevel(factor(group), ref = "ICTR")
  )


# ---- Baseline correlation check ----------------------------------------------
# Examine pairwise correlations between outcomes at baseline.
# Moderate correlations support a joint multivariate model that
# estimates residual covariance

nmb_data %>%
  filter(visit == "2") %>%
  select(score_depression, score_stress, score_symptoms) %>%
  tab_corr(
    corr.method = "spearman",
    triangle    = "upper",
    show.p      = TRUE
  )

# ---- Z-score outcomes --------------------------------------------------------
# Stan prefers numerically stable inputs, Z-scoring (mean 0, SD 1) avoids convergence issues when outcomes are on very different scales and makes
# regression coefficients directly comparable across response variables.
# The [, 1] index extracts the numeric vector from scale()'s matrix output.

nmb_data <- nmb_data %>%
  mutate(
    score_depression_z = scale(score_depression)[, 1],
    score_stress_z     = scale(score_stress)[, 1],
    score_symptoms_z   = scale(score_symptoms)[, 1]
  )


# ---- Data preparation for models ---------------------------------------------
# each participant's baseline z-score to all rows as a covariate.
# Filter out the baseline visit — it is now carried as a covariate, not a
# response row (same logic as the lmer/ANCOVA scripts).
# visit is coded numeric to estimate a linear time trend across follow-up
# visits; change to as.factor(visit) for a categorical time effect (not always practical with small sample size, dilutes effect)

model_data <- nmb_data %>%
  group_by(id) %>%
  mutate(
    bl_depression_z = score_depression_z[visit == "2"],
    bl_stress_z     = score_stress_z[visit == "2"],
    bl_symptoms_z   = score_symptoms_z[visit == "2"]
  ) %>%
  filter(visit != "2") %>%
  ungroup() %>%
  mutate(
    visit = as.numeric(as.character(visit)),
    sex   = as.factor(sex)
  )


# ---- Prior specification -----------------------------------------------------
# Weakly informative priors provide mild regularisation while allowing the
# data to drive the posterior. All outcomes are z-scored so priors are on a
# standardised scale (expected coefficients in the range ±1-2 SD units).
#
# Coefficient priors — student_t(4, 0, 1):
#   Heavier tails than Normal; robust to occasional large effects (mostly due to noise/rtm/measurement error).
#   Centred at 0 (no a priori direction assumed for group differences).
#   Adjust the scale (third argument) if outcomes are not z-scored.
#
# Intercept priors — normal(0, 1):
#   Centred at 0 on the z-scored scale. Shift the mean slightly (e.g., -0.25)
#   if prior literature suggests a directional baseline expectation.
#
# Sigma / SD priors — exponential(1):
#   Places most mass near 0 with a long tail; appropriate for SDs of z-scored
#   data where residual SDs are expected near 1.
#
# Correlation matrix — lkj(2):
#   Mild regularisation toward zero correlation; use lkj(1) for a uniform
#   prior over all correlation matrices (less regularisation).
#
# NOTE on resp identifiers: brms derives the resp= label by removing
# underscores from the variable name, e.g. score_depression_z → scoredepressionz.
# Verify labels by running prior_summary(mood_mv) after fitting.

priors <- c(
  # Regression coefficients
  prior(student_t(4, 0, 1), class = "b",         resp = "scoredepressionz"),
  prior(student_t(4, 0, 1), class = "b",         resp = "scorestressz"),
  prior(student_t(4, 0, 1), class = "b",         resp = "scoresymptomsz"),

  # Intercepts
  prior(normal(0, 1),       class = "Intercept", resp = "scoredepressionz"),
  prior(normal(0, 1),       class = "Intercept", resp = "scorestressz"),
  prior(normal(0, 1),       class = "Intercept", resp = "scoresymptomsz"),

  # Residual SDs
  prior(exponential(1),     class = "sigma",     resp = "scoredepressionz"),
  prior(exponential(1),     class = "sigma",     resp = "scorestressz"),
  prior(exponential(1),     class = "sigma",     resp = "scoresymptomsz"),

  # Random effect SDs
  prior(exponential(1),     class = "sd",        resp = "scoredepressionz"),
  prior(exponential(1),     class = "sd",        resp = "scorestressz"),
  prior(exponential(1),     class = "sd",        resp = "scoresymptomsz"),

  # Residual correlation matrix between outcomes
  prior(lkj(2),             class = "rescor")
)


# ---- Fit multivariate Bayesian model -----------------------------------------
# mvbind() specifies a multivariate model: each response shares the same
# right-hand-side formula but has its own intercept, sigma, SD, and priors.
# rescor = TRUE (default) estimates residual correlations between outcomes via
# the LKJ prior above.
#
# Sampling settings:
#   iter = 4000 / warmup = 1500 — more iterations recommended for multivariate
#     models; increase if Rhat > 1.01 or bulk/tail ESS < 400.
#   adapt_delta = 0.99 — reduces divergent transitions; increase toward 0.999
#     if divergences persist after refitting.
#   backend = "cmdstanr" — faster and more memory-efficient than rstan.
#   seed — set for reproducibility.
#
# Prior predictive check:
#   Uncomment sample_prior = "only" to draw samples from the prior alone
#   (no likelihood). Inspect pp_check() output to verify that priors produce
#   plausible outcome ranges before seeing the data. The prior predictive
#   distribution should cover realistic values without being implausibly wide.
#   Remove or comment out for the final model run.

mood_mv <- brm(
  mvbind(score_depression_z, score_stress_z, score_symptoms_z) ~
    group + visit + age + sex +
    bl_depression_z + bl_stress_z + bl_symptoms_z + (1 | id),
  data    = model_data,
  prior   = priors,
  # sample_prior = "only",   # uncomment for prior predictive check
  chains  = 4,
  cores   = 2,
  iter    = 4000,
  warmup  = 1500,
  control = list(adapt_delta = 0.99),
  backend = "cmdstanr",
  refresh = 100,
  silent  = 2,
  seed    = 123
)

summary(mood_mv)

# Verify that resp= labels match the fitted model
prior_summary(mood_mv)

# Optional: save workspace after fitting — model sampling is slow
# save.image(file = "bayes_mood_results.RData")


# ---- Posterior summary -------------------------------------------------------
# describe_posterior() returns median, HDI credible intervals, probability of
# direction (pd), and Bayes factors (bf) for each parameter.
# pd > 0.95 indicates the effect is likely in the estimated direction;
# pd > 0.99 is considered strong directional evidence.
# HDI (Highest Density Interval) is preferred over equal-tailed CI for
# asymmetric posteriors.

table_posterior <- describe_posterior(
  mood_mv,
  centrality = "median",
  ci_method  = "HDI",
  test       = c("pd", "bf")
)

print_html(table_posterior)


# ---- MCMC Diagnostics --------------------------------------------------------
# Always run before interpreting results.
# Convergence criteria: Rhat < 1.01 and bulk/tail ESS > 400 for all parameters.
# Divergent transitions signal sampler geometry problems — increase adapt_delta
# or consider reparameterisation if they persist.

# Trace plots: inspect chain mixing and stationarity for intervention params
mcmc_trace(mood_mv,
           regex_pars  = "b_.*group",
           facet_args  = list(ncol = 3))

# Trace with NUTS diagnostic overlay (highlights divergent transitions)
mcmc_trace(mood_mv,
           regex_pars = "b_.*group",
           np         = nuts_params(mood_mv))

# Posterior density areas for intervention group coefficients
# Inner band = 50% CI; outer band = 95% CI
mcmc_areas(mood_mv,
           regex_pars = "b_scoredepressionz_group",
           prob       = 0.50,
           prob_outer = 0.95) +
  geom_vline(xintercept = 0, linetype = "dashed")

# Posterior predictive checks: observed vs replicated outcome distributions
# "dens_overlay_grouped" overlays pp-replicated densities per intervention arm.
# Systematic misfit suggests the model family or priors need revision.
pp_check(mood_mv, resp = "scoredepressionz",
         type   = "dens_overlay_grouped",
         group  = "group",
         ndraws = 25) +
  ggtitle("Depression: posterior predictive check by group")

pp_check(mood_mv, resp = "scorestressz",
         type   = "dens_overlay_grouped",
         group  = "group",
         ndraws = 25) +
  ggtitle("Stress: posterior predictive check by group")

pp_check(mood_mv, resp = "scoresymptomsz",
         type   = "dens_overlay_grouped",
         group  = "group",
         ndraws = 25) +
  ggtitle("Symptoms: posterior predictive check by group")


# ---- Half-eye plot: intervention coefficients (simple) -----------------------
# gather_draws() extracts all posterior samples for parameters matching the
# regex. stat_halfeye() from ggdist displays the posterior as a half-density
# + interval plot. .width = c(0.50, 0.95) draws 50% and 95% credible intervals.
# The dashed line at 0 is the null reference (no group difference).

posterior_draws <- mood_mv %>%
  gather_draws(`b_.*group.*`, regex = TRUE) %>%
  mutate(
    # Map brms parameter names to readable outcome labels
    outcome = case_when(
      grepl("scoredepressionz", .variable) ~ "Depression",
      grepl("scorestressz",     .variable) ~ "Stress",
      grepl("scoresymptomsz",   .variable) ~ "Symptoms"
    ),
    # Map group codes to display labels
    group = case_when(
      grepl("ICOMB", .variable) ~ "Combined",
      grepl("IFER",  .variable) ~ "Fermented",
      grepl("IFIB",  .variable) ~ "High Fibre"
    )
  )

plot_halfeye <- ggplot(posterior_draws, aes(x = .value, y = group)) +
  stat_halfeye(.width = c(0.50, 0.95)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "red") +
  facet_wrap(~outcome, nrow = 3) +
  labs(x = "Standardised Coefficient", y = "Intervention Group") +
  theme_minimal()

plot_halfeye

ggsave("plots/bayes_mood_halfeye.png", plot_halfeye,
       width = 4, height = 6, dpi = 300)


# ---- Emmeans contrast half-eye plots by visit if needed --------------------------------
# emmeans() marginalises over the posterior predictive distribution at each
# visit × group combination (re_formula = NA excludes random effects, giving
# population-level estimates). contrast() computes pairwise differences vs.
# the control arm. gather_emmeans_draws() converts to tidy long-format draws.
# Separate plots per visit are combined side-by-side with patchwork.

emm_depression <- emmeans(mood_mv, ~ group | visit,
                          resp = "scoredepressionz", re_formula = NA)
emm_stress     <- emmeans(mood_mv, ~ group | visit,
                          resp = "scorestressz",     re_formula = NA)
emm_symptoms   <- emmeans(mood_mv, ~ group | visit,
                          resp = "scoresymptomsz",   re_formula = NA)

contr_depression <- contrast(emm_depression, "trt.vs.ctrl", ref = "ICTR")
contr_stress     <- contrast(emm_stress,     "trt.vs.ctrl", ref = "ICTR")
contr_symptoms   <- contrast(emm_symptoms,   "trt.vs.ctrl", ref = "ICTR")

# Convert to tidy draws and label by outcome
draws_depression <- gather_emmeans_draws(contr_depression) %>% mutate(outcome = "Depression")
draws_stress     <- gather_emmeans_draws(contr_stress)     %>% mutate(outcome = "Stress")
draws_symptoms   <- gather_emmeans_draws(contr_symptoms)   %>% mutate(outcome = "Symptoms")

# Combine and recode contrast labels for display
draws_all <- bind_rows(draws_depression, draws_stress, draws_symptoms) %>%
  mutate(
    contrast = recode(
      contrast,
      "IFIB - ICTR"  = "High Fibre",
      "IFER - ICTR"  = "Fermented",
      "ICOMB - ICTR" = "Combined"
    )
  )

# Generic half-eye plot function for a single visit timepoint.
# Adapt visit_code to your numeric visit values and visit_label to your
# study's timepoint names.
halfeye_visit <- function(draws, visit_code, visit_label) {
  draws %>%
    filter(visit == visit_code) %>%
    ggplot(aes(x = .value, y = contrast)) +
    stat_halfeye(.width = c(0.50, 0.95)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "red") +
    facet_wrap(~outcome, nrow = 3) +
    labs(x = "Standardised Coefficient", y = "Intervention Group") +
    ggtitle(visit_label) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}

# Generate one panel per visit (adapt visit codes and labels as needed)
plot_v3 <- halfeye_visit(draws_all, visit_code = 3, visit_label = "Week 4")

plot_v4 <- halfeye_visit(draws_all, visit_code = 4, visit_label = "Week 8") +
  theme(axis.title.y = element_blank(),
        axis.text.y  = element_blank())

combined_visits <- plot_v3 | plot_v4

combined_visits

ggsave("plots/bayes_mood_by_visit.png", combined_visits,
       width = 8, height = 6, dpi = 300)


# ---- Sensitivity analysis: weak priors ---------------------------------------
# Refit the model with diffuse priors to verify that conclusions are not
# driven by prior choice. If median estimates and credible intervals are
# similar across prior specifications, results are robust. Large differences
# (> ~20-30% change) indicate the data are sparse and prior choice matters a bit too much.
#
# Weak priors used here:
#   normal(0, 5) for coefficients and intercepts — nearly uninformative on
#     the z-scored scale (allows very large effects up to ±10 SD, essentially unconstrained)
#   exponential(0.5) for SDs — more diffuse than the main model
#   lkj(1) for the correlation matrix — uniform over all valid matrices

weak_priors <- c(
  prior(normal(0, 5),     class = "b",         resp = "scoredepressionz"),
  prior(normal(0, 5),     class = "b",         resp = "scorestressz"),
  prior(normal(0, 5),     class = "b",         resp = "scoresymptomsz"),

  prior(normal(0, 5),     class = "Intercept", resp = "scoredepressionz"),
  prior(normal(0, 5),     class = "Intercept", resp = "scorestressz"),
  prior(normal(0, 5),     class = "Intercept", resp = "scoresymptomsz"),

  prior(exponential(0.5), class = "sigma",     resp = "scoredepressionz"),
  prior(exponential(0.5), class = "sigma",     resp = "scorestressz"),
  prior(exponential(0.5), class = "sigma",     resp = "scoresymptomsz"),

  prior(exponential(0.5), class = "sd",        resp = "scoredepressionz"),
  prior(exponential(0.5), class = "sd",        resp = "scorestressz"),
  prior(exponential(0.5), class = "sd",        resp = "scoresymptomsz"),

  prior(lkj(1),           class = "rescor")
)

# Fewer chains and iterations — sensitivity check only, not for inference
mood_mv_weak <- brm(
  mvbind(score_depression_z, score_stress_z, score_symptoms_z) ~
    group + visit + age + sex +
    bl_depression_z + bl_stress_z + bl_symptoms_z + (1 | id),
  data    = model_data,
  prior   = weak_priors,
  chains  = 2,
  iter    = 1000,
  cores   = 2,
  backend = "cmdstanr",
  seed    = 123
)

# Extract and compare intervention parameter estimates between prior specs
orig_results <- describe_posterior(mood_mv) %>%
  filter(grepl("group", Parameter)) %>%
  select(Parameter, Median, CI_low, CI_high) %>%
  mutate(Model = "Informative Priors")

weak_results <- describe_posterior(mood_mv_weak) %>%
  filter(grepl("group", Parameter)) %>%
  select(Parameter, Median, CI_low, CI_high) %>%
  mutate(Model = "Weak Priors")

sensitivity_comparison <- bind_rows(orig_results, weak_results)

# Quantify % change in median estimates — large values flag prior sensitivity
sensitivity_comparison %>%
  select(Parameter, Model, Median) %>%
  pivot_wider(names_from = Model, values_from = Median) %>%
  mutate(
    Difference     = `Informative Priors` - `Weak Priors`,
    Percent_Change = abs(Difference / `Weak Priors`) * 100
  )
