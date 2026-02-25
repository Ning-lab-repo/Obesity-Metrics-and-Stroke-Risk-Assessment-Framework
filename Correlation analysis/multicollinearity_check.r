

# ===============================
# 成年人体格测量数据清洗
# 将生理学不可能的值转换为NA
# ===============================

# 假设数据框为 df
# df <- read.csv("your_data.csv")

# ===============================
# 1. 定义成年人生理学合理范围
# ===============================

# 身高范围 (cm)
height_min <- 130  # 成年人最低身高（极矮身材）
height_max <- 220  # 成年人最高身高（姚明身高226cm，留余地）

# 体重范围 (kg)
weight_min <- 30   # 成年人最低体重（严重营养不良下限）
weight_max <- 250  # 成年人最高体重（病态肥胖上限）

# 腰围范围 (cm)
wc_min <- 40       # 腰围最小值（极瘦）
wc_max <- 200      # 腰围最大值（极度肥胖）

# 臀围范围 (cm)
hc_min <- 50       # 臀围最小值（极瘦）
hc_max <- 200      # 臀围最大值（极度肥胖）

# ===============================
# 2. 清洗前统计
# ===============================

cat("===============================\n")
cat("数据清洗报告 - 成年人体格测量\n")
cat("===============================\n\n")

cat("【清洗前统计】\n")
cat(sprintf("总样本量: %d\n", nrow(df)))
cat(sprintf("Height 缺失: %d (%.2f%%)\n", 
            sum(is.na(df$Height)), 
            sum(is.na(df$Height))/nrow(df)*100))
cat(sprintf("Weight 缺失: %d (%.2f%%)\n", 
            sum(is.na(df$Weight)), 
            sum(is.na(df$Weight))/nrow(df)*100))
cat(sprintf("WC 缺失: %d (%.2f%%)\n", 
            sum(is.na(df$WC)), 
            sum(is.na(df$WC))/nrow(df)*100))
cat(sprintf("HC 缺失: %d (%.2f%%)\n", 
            sum(is.na(df$HC)), 
            sum(is.na(df$HC))/nrow(df)*100))

# ===============================
# 3. 识别异常值
# ===============================

cat("\n【识别异常值】\n")

# Height 异常值
height_outliers <- sum(!is.na(df$Height) & 
                       (df$Height < height_min | df$Height > height_max))
if (height_outliers > 0) {
  cat(sprintf("⚠️  Height 异常值: %d 个\n", height_outliers))
  cat(sprintf("   范围: %.1f - %.1f (合理范围: %d - %d)\n", 
              min(df$Height, na.rm = TRUE), 
              max(df$Height, na.rm = TRUE),
              height_min, height_max))
  
  # 显示异常值示例
  abnormal_heights <- df$Height[!is.na(df$Height) & 
                                (df$Height < height_min | df$Height > height_max)]
  cat(sprintf("   异常值示例: %s\n", 
              paste(head(sort(abnormal_heights), 5), collapse = ", ")))
}

# Weight 异常值
weight_outliers <- sum(!is.na(df$Weight) & 
                       (df$Weight < weight_min | df$Weight > weight_max))
if (weight_outliers > 0) {
  cat(sprintf("⚠️  Weight 异常值: %d 个\n", weight_outliers))
  cat(sprintf("   范围: %.1f - %.1f (合理范围: %d - %d)\n", 
              min(df$Weight, na.rm = TRUE), 
              max(df$Weight, na.rm = TRUE),
              weight_min, weight_max))
  
  abnormal_weights <- df$Weight[!is.na(df$Weight) & 
                                (df$Weight < weight_min | df$Weight > weight_max)]
  cat(sprintf("   异常值示例: %s\n", 
              paste(head(sort(abnormal_weights), 5), collapse = ", ")))
}

# WC 异常值
wc_outliers <- sum(!is.na(df$WC) & 
                   (df$WC < wc_min | df$WC > wc_max))
if (wc_outliers > 0) {
  cat(sprintf("⚠️  WC 异常值: %d 个\n", wc_outliers))
  cat(sprintf("   范围: %.1f - %.1f (合理范围: %d - %d)\n", 
              min(df$WC, na.rm = TRUE), 
              max(df$WC, na.rm = TRUE),
              wc_min, wc_max))
}

# HC 异常值
hc_outliers <- sum(!is.na(df$HC) & 
                   (df$HC < hc_min | df$HC > hc_max))
if (hc_outliers > 0) {
  cat(sprintf("⚠️  HC 异常值: %d 个\n", hc_outliers))
  cat(sprintf("   范围: %.1f - %.1f (合理范围: %d - %d)\n", 
              min(df$HC, na.rm = TRUE), 
              max(df$HC, na.rm = TRUE),
              hc_min, hc_max))
}

# ===============================
# 4. 将异常值转换为NA
# ===============================

cat("\n【执行清洗】\n")

# 创建备份
df_original <- df

# Height: 130-220 cm
df$Height[!is.na(df$Height) & 
          (df$Height < height_min | df$Height > height_max)] <- NA
cat(sprintf("✓ Height: %d 个异常值转为NA\n", height_outliers))

# Weight: 30-250 kg
df$Weight[!is.na(df$Weight) & 
          (df$Weight < weight_min | df$Weight > weight_max)] <- NA
cat(sprintf("✓ Weight: %d 个异常值转为NA\n", weight_outliers))

# WC: 40-200 cm
df$WC[!is.na(df$WC) & 
      (df$WC < wc_min | df$WC > wc_max)] <- NA
cat(sprintf("✓ WC: %d 个异常值转为NA\n", wc_outliers))

# HC: 50-200 cm
df$HC[!is.na(df$HC) & 
      (df$HC < hc_min | df$HC > hc_max)] <- NA
cat(sprintf("✓ HC: %d 个异常值转为NA\n", hc_outliers))
