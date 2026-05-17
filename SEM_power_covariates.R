
#===========================================#
#Simulations for SEM with basic covariates
#===========================================#

library(lavaan)
library(dplyr)
library(tidyr)
library(ggplot2)

set.seed(2025)

# Simulation parameters

replications <- 1000
sample_sizes <- c(400, 600, 800, 1000) 
results <- list()

# Population model with covariates and labeled paths

#Factor loadings and path coefficients can be changed as needed

population_model <- '
  # Latent factors
  DietHealthy =~ 0.5*d1 + 0.5*d2 + 0.4*d3 + 0.4*d4 + 0.3*d5 + 0.3*d6 
  Mental      =~ 0.7*m1 + 0.7*m2 + 0.7*m3
  Cog         =~ 0.7*c1 + 0.7*c2 + 0.7*c3

  # Structural paths
  Mental ~ 0.2*a*DietHealthy
  Cog    ~ 0.2*b*Mental + 0.2*c*DietHealthy

  # Covariate effects
  Mental ~ 0.1*Sex + 0.1*Age + 0.1*IPAQ + 0.1*SES
  Cog    ~ 0.1*Sex + 0.1*Age + 0.1*IPAQ + 0.1*SES
  DietHealthy ~ 0.1*Sex + 0.1*Age + 0.1*IPAQ + 0.1*SES

  # Latent variances
  DietHealthy ~~ 1*DietHealthy
  Mental      ~~ 1*Mental
  Cog         ~~ 1*Cog
'

# Fitted model (has to be the same as population, with labels for indirect/total)

fitted_model <- '
  # Latent factors
  DietHealthy =~ d1 + d2 + d3 + d4 + d5 + d6
  Mental      =~ m1 + m2 + m3
  Cog         =~ c1 + c2 + c3

  # Structural paths with labels
  Mental ~ a*DietHealthy
  Cog    ~ b*Mental + c*DietHealthy

  # Covariates
  Mental ~ Sex + Age + IPAQ + SES
  Cog    ~ Sex + Age + IPAQ + SES
  DietHealthy ~ Sex + Age + IPAQ + SES

  # Indirect and total effects
  ind := a*b
  total := c + (a*b)
'

# MC simulation loop (messages added to check simulation status) 

power_results <- list()

for(N in sample_sizes){
  est_list <- list()
  
  cat("Simulating sample size N =", N, "...\n")
  
  for(i in 1:replications){
    # Generate covariates
    Sex <- rbinom(N, 1, 0.5)
    Age <- rnorm(N, 36, 5) 
    Age[Age < 18 | Age > 60] <- 30 #to replace out of range
    IPAQ <- sample(1:3, N, replace = TRUE, prob = c(0.30, 0.45, 0.25)) #physical actvity categories
    SES <- sample(1:4, N, replace = TRUE) # 4 brackets
    
    # Simulate data
    dat <- simulateData(population_model, sample.nobs = N)
    dat$Sex <- Sex
    dat$Age <- Age
    dat$IPAQ <- IPAQ
    dat$SES <- SES
    
    # Fit model
    
    fit <- try(sem(fitted_model, data = dat, std.lv = TRUE), silent = TRUE)
    #fit <- try(sem(fitted_model, data = dat, std.lv = TRUE, estimator = "MLR"), silent = TRUE)
    
    if(!inherits(fit, "try-error")){
      sol <- standardizedSolution(fit)
      est_list[[i]] <- sol
    }
    
  }
  
  # Compute power (proportion of significant paths)
  
  paths <- c("a", "b", "c", "ind", "total")
  power_vec <- sapply(paths, function(p){
    sig <- sapply(est_list, function(x){
      row <- x[x$label == p, ]
      if(nrow(row)==0) return(NA)
      abs(row$z) > 1.96  # p < 0.05 from the z-value
    })
    mean(sig, na.rm = TRUE)
  })
  
  power_results[[as.character(N)]] <- power_vec
}

#save and reload data if needed

#saveRDS(power_results, file = "outputs/power_results_mediation_weak.rds")

#power_results <- readRDS("outputs/power_results_mediation.rds")

# Convert to long format for plotting 

power_df <- do.call(rbind, lapply(names(power_results), function(n){
  data.frame(N = as.numeric(n), t(power_results[[n]]))
}))
colnames(power_df) <- c("N", "a", "b", "c", "ind", "total")

power_long <- power_df %>%
  pivot_longer(-N, names_to = "path", values_to = "power") %>%
  mutate(path = recode(path,
                       a="Mental ~ DietHealthy (a)",
                       b="Cog ~ Mental (b)",
                       c="Cog ~ DietHealthy (c)",
                       ind="Indirect Diet → Mental → Cog",
                       total="Total effect"))

View(power_long) # to check numbers first

# plot

power_plot <- ggplot(power_long, aes(x = N, y = power, color = path)) +
  geom_line(linewidth = 1.2) +
  geom_point(size=2) +
  geom_hline(yintercept = 0.8, linetype="dashed", color="red") + # 80% power
  geom_hline(yintercept = 0.9, linetype="dashed", color="blue") + # 90% power
  scale_y_continuous(labels = scales::percent_format(accuracy=1), limits=c(0,1)) +
  labs(title="Estimated Power vs Sample Size (Paths + Indirect + Total)",
       x="Sample Size (N)",
       y="Power",
       color="Path") +
  theme_minimal()


# --- Plot model structure and paths using the last fitted model ---
if (exists("fit") && inherits(fit, "lavaan")) {
  library(semPlot)
  semPaths(
    fit,
    what = "std",
    layout = "spring",
    style = "mx",
    edge.label.cex = 1,
    sizeLat = 8,
    sizeMan = 6,
    residuals = FALSE,
    intercepts = FALSE
  
  )
}

