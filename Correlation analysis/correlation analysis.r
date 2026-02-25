# ============================================================
# Obesity Index Correlation & Multicollinearity Analysis
# ============================================================

rm(list = ls())
options(stringsAsFactors = FALSE)

# ------------------------------
# Load packages
# ------------------------------
library(tidyverse)
library(psych)
library(corrplot)
# ------------------------------
# Paths
# ------------------------------
data_path   <- "data/stroke_3839.csv"
output_dir  <- "output/correlation"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------
# Load data
# ------------------------------
k_data <- read.csv(data_path, fileEncoding = "UTF-8")

# ------------------------------
# Obesity variables
# ------------------------------
obesity_vars <- c(
  "Height","Weight","WC","HC","BMI","WHR",
  "WHtR","HHtR","ABSI","BRI","BF_percent"
)

obesity_data <- k_data %>%
  select(all_of(obesity_vars)) %>%
  drop_na()

cat("Sample size (complete cases):", nrow(obesity_data), "\n")

# ============================================================
# 1. Descriptive statistics
# ============================================================

desc_stats <- psych::describe(obesity_data)

write.csv(
  round(desc_stats[,c("n","mean","sd","min","max")],2),
  file.path(output_dir,"descriptive_statistics.csv")
)

# ============================================================
# 2. Correlation matrix
# ============================================================

cor_matrix <- cor(obesity_data, method = "pearson")
cor_test   <- psych::corr.test(obesity_data)

write.csv(round(cor_matrix,4),
          file.path(output_dir,"correlation_matrix.csv"))

write.csv(round(cor_test$p,4),
          file.path(output_dir,"correlation_pvalues.csv"))

# Heatmap
pdf(file.path(output_dir,"correlation_heatmap.pdf"),
    width = 10, height = 8)

corrplot(cor_matrix,
         method = "color",
         type = "upper",
         order = "hclust",
         addCoef.col = "black",
         tl.srt = 45)

dev.off()

# ============================================================
# Table 1: Stroke vs Non-stroke Comparison
# ============================================================

rm(list = ls())
options(stringsAsFactors = FALSE)

library(tidyverse)

# ------------------------------
# Paths
# ------------------------------
data_path  <- "data/stroke_3839.csv"
output_dir <- "output/table1"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------
# Load data
# ------------------------------
k_data <- read.csv(data_path, fileEncoding = "UTF-8")

# ------------------------------
# Variable lists
# ------------------------------
continuous_vars <- c(
  "age","Height","Weight","WC","HC","BMI","WHR",
  "Heart_rate","Sbp","Dbp",
  "HDL_cholesterol","LDL_direct","Cholesterol",
  "Triglycerides","Glucose",
  "WHtR","HHtR","ABSI","BRI","BF_percent"
)

categorical_vars <- c(
  "sex","Smoke","Alcohol","DM",
  "thrombosis","lacunar","ICH"
)

# ============================================================
# Continuous variables
# ============================================================

cont_table <- lapply(continuous_vars, function(v){

  df <- k_data %>%
    select(subject, all_of(v)) %>%
    drop_na()

  g0 <- df %>% filter(subject==0) %>% pull(v)
  g1 <- df %>% filter(subject==1) %>% pull(v)

  if(length(g0)<2 | length(g1)<2) return(NULL)

  t_res <- t.test(g0,g1)

  tibble(
    Variable = v,
    Non_stroke = sprintf("%.2f ± %.2f",mean(g0),sd(g0)),
    Stroke     = sprintf("%.2f ± %.2f",mean(g1),sd(g1)),
    P_value    = signif(t_res$p.value,3),
    Type       = "Continuous"
  )
})

cont_table <- bind_rows(cont_table)

# ============================================================
# Categorical variables
# ============================================================

cat_table <- lapply(categorical_vars, function(v){

  df <- k_data %>%
    select(subject, all_of(v)) %>%
    drop_na()

  tab <- table(df[[v]], df$subject)

  if(any(tab<5)){
    p_val <- fisher.test(tab)$p.value
  }else{
    p_val <- chisq.test(tab)$p.value
  }

  fmt <- function(x,total){
    sprintf("%d (%.1f%%)",x,100*x/total)
  }

  tibble(
    Variable = v,
    Non_stroke = fmt(tab["1","0"],sum(tab[,"0"])),
    Stroke     = fmt(tab["1","1"],sum(tab[,"1"])),
    P_value    = signif(p_val,3),
    Type       = "Categorical"
  )
})

cat_table <- bind_rows(cat_table)

# ============================================================
# Combine
# ============================================================

table1 <- bind_rows(cont_table,cat_table)

write.csv(table1,
          file.path(output_dir,"table1_results.csv"),
          row.names = FALSE)

cat("Table 1 analysis completed.\n")