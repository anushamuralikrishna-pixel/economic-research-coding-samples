# Treatment Effects Analysis
#
# Context:
# This script analyzes the short- and long-term effects of three treatment
# programs on household outcomes. It estimates treatment-specific effects,
# constructs an indicator for receiving any treatment, and tests whether the
# short-term treatment effect varies with baseline literacy.
#
# The analysis demonstrates data manipulation with dplyr, linear regression,
# interaction terms, and regression-table generation.
#
# Required input:
# Place "Dataset1 (1).csv" in the same folder as this script.

library(dplyr)
library(stargazer)


# Load data ---------------------------------------------------------------

treatment_data <- read.csv("Dataset1 (1).csv")


# 1. Short-term treatment effects ----------------------------------------

short_term_data <- treatment_data |>
  filter(round == 1)

short_term_behavior_model <- lm(
  fam_asb_st ~ cashassonly + tpassonly + tpcashass,
  data = short_term_data
)

short_term_consumption_model <- lm(
  cstot2wusd_stav ~ cashassonly + tpassonly + tpcashass,
  data = short_term_data
)

summary(short_term_behavior_model)
summary(short_term_consumption_model)

stargazer(
  short_term_behavior_model,
  short_term_consumption_model,
  type = "text",
  title = "Short-Term Treatment Effects",
  out = "short_term_results.txt"
)


# 2. Long-term treatment effects -----------------------------------------

long_term_data <- treatment_data |>
  filter(round == 5)

long_term_behavior_model <- lm(
  fam_asb_lt ~ cashassonly + tpassonly + tpcashass,
  data = long_term_data
)

long_term_consumption_model <- lm(
  cstot2wusd_ltav ~ cashassonly + tpassonly + tpcashass,
  data = long_term_data
)

summary(long_term_behavior_model)
summary(long_term_consumption_model)

stargazer(
  long_term_behavior_model,
  long_term_consumption_model,
  type = "text",
  title = "Long-Term Treatment Effects",
  out = "long_term_results.txt"
)


# 3. Effect of receiving any treatment -----------------------------------

treatment_data <- treatment_data |>
  mutate(
    any_treatment = if_else(
      cashassonly == 1 | tpassonly == 1 | tpcashass == 1,
      1,
      0
    )
  )

short_term_data <- treatment_data |>
  filter(round == 1)

any_treatment_model <- lm(
  fam_asb_st ~ any_treatment,
  data = short_term_data
)

summary(any_treatment_model)

stargazer(
  any_treatment_model,
  type = "text",
  title = "Short-Term Effect of Receiving Any Treatment",
  out = "any_treatment_results.txt"
)


# 4. Treatment-effect heterogeneity by baseline literacy -----------------

short_term_data <- short_term_data |>
  mutate(
    literate = if_else(literacy_b == 2, 1, 0)
  )

literacy_interaction_model <- lm(
  fam_asb_st ~ any_treatment * literate,
  data = short_term_data
)

summary(literacy_interaction_model)

stargazer(
  literacy_interaction_model,
  type = "text",
  title = "Short-Term Treatment Effects by Baseline Literacy",
  out = "literacy_interaction_results.txt"
)