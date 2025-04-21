# ====================================================
# Projeto 1VA - Aprendizado de Máquina com SVM em R
# Autor: Ian Lucas Almeida
# Curso: Mestrado em Informática Aplicada - UFRPE/PPGIA - 2025 | CPAD
# Data: Abril de 2025
# Objetivo: Prever evasão estudantil com SVM
# ====================================================

# ----------------------------
# 1. Instalação e carregamento de pacotes
# ----------------------------
required_packages <- c("dplyr", "ggplot2", "readr", "tibble", "e1071", "caret", 
                       "corrplot", "parallel", "doParallel", "pROC", "janitor")

invisible(lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg, dependencies = TRUE)
  library(pkg, character.only = TRUE)
}))

# ----------------------------
# 2. Função auxiliar: moda
# ----------------------------
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# ----------------------------
# 3. Carregamento e limpeza dos dados
# ----------------------------
url_csv <- "https://raw.githubusercontent.com/ianlucasalmeida/projeto_1va-SVM/refs/heads/main/dataset/Student_Dropout_rate.csv"

dataset <- read_csv2(url_csv) %>%
  clean_names() %>%  # <- padroniza os nomes das colunas
  mutate(target = recode(target, "Graduate" = 0, "Dropout" = 1, "Enrolled" = 2)) %>%
  filter(target != 2)

# ----------------------------
# 4. Análise Exploratória
# ----------------------------
cat("Total de amostras:", nrow(dataset), "\n")
cat("Taxa de evasão:", round(mean(dataset$target) * 100, 2), "%\n")

if (!dir.exists("plots")) dir.create("plots")

ggplot(dataset, aes(x = age_at_enrollment, fill = factor(target))) +
  geom_histogram(binwidth = 2, position = "stack") +
  labs(title = "Distribuição de Idades por Status", x = "Idade na Matrícula", y = "Contagem") +
  theme_minimal()
ggsave("plots/idade_distribuicao.png", width = 8, height = 5)

dropout_ages <- dataset %>% filter(target == 1) %>% pull(age_at_enrollment)
cat("Idade mais comum entre evasores:", Mode(dropout_ages), "anos\n")
cat("Média de idade dos evasores:", round(mean(dropout_ages), 1), "anos\n")

numeric_cols <- sapply(dataset, is.numeric)
dataset_numeric <- dataset[, numeric_cols]
correlations <- cor(dataset_numeric, use = "complete.obs")["target", ]
cor_df <- data.frame(variavel = names(correlations), correlacao = correlations) %>%
  arrange(desc(correlacao))
print(head(cor_df, 10))
print(tail(cor_df, 5))

png("plots/heatmap_correlacoes.png", width = 800, height = 600)
corrplot(cor(dataset_numeric, use = "complete.obs"), method = "color", type = "upper",
         tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.7)
dev.off()

if ("gender" %in% names(dataset)) {
  ggplot(dataset, aes(x = gender, fill = factor(target))) +
    geom_bar(position = "fill") +
    labs(title = "Proporção de Evasão por Gênero", x = "Gênero", y = "Proporção") +
    theme_minimal()
  ggsave("plots/evasao_genero.png", width = 8, height = 5)
}

if ("curricular_units_1st_sem_grade" %in% names(dataset)) {
  ggplot(dataset, aes(x = curricular_units_1st_sem_grade, y = target)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "glm", method.args = list(family = "binomial")) +
    labs(title = "Nota x Evasão", x = "Nota", y = "Evasão") +
    theme_minimal()
  ggsave("plots/notas_evasao.png", width = 8, height = 5)
}

# ----------------------------
# 5. Pré-processamento
# ----------------------------
X <- dataset %>% select(-target)
y <- factor(dataset$target)
preproc <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preproc, X)
dataset_scaled <- bind_cols(as_tibble(X_scaled), target = y)

# ----------------------------
# 6. Treinamento inicial do SVM
# ----------------------------
svm_initial <- svm(target ~ ., data = dataset_scaled, kernel = "linear")
sv <- as.matrix(svm_initial$SV)
coefs <- as.matrix(svm_initial$coefs)
importances <- colSums(abs(sv * matrix(coefs, nrow = nrow(sv), ncol = ncol(sv), byrow = TRUE)))
names(importances) <- colnames(X_scaled)
top_features <- names(sort(importances, decreasing = TRUE))[1:10]

# ----------------------------
# 7. Avaliação com múltiplas combinações
# ----------------------------
combos <- unlist(lapply(1:min(3, length(top_features)),
                        function(r) combn(top_features, r, simplify = FALSE)), recursive = FALSE)

kernels <- c("linear", "radial", "polynomial")
combo_kernel_pairs <- expand.grid(combo = I(combos), kernel = kernels)

process_combination <- function(args) {
  cols <- args$combo
  kernel <- args$kernel
  df <- dataset_scaled %>% select(-all_of(cols))
  X <- df %>% select(-target)
  y <- df$target
  set.seed(5)
  idx <- createDataPartition(y, p = 0.8, list = FALSE)
  train <- bind_cols(X[idx, ], target = y[idx])
  test <- bind_cols(X[-idx, ], target = y[-idx])
  model <- svm(target ~ ., data = train, kernel = kernel, gamma = "auto")
  pred <- predict(model, test %>% select(-target))
  acc <- mean(pred == test$target) * 100
  list(accuracy = acc, kernel = kernel, cols = cols, pred = pred, test = test)
}

cl <- makeCluster(2)
registerDoParallel(cl)
clusterExport(cl, c("dataset_scaled", "process_combination", "createDataPartition", "svm", "predict"))
clusterEvalQ(cl, { library(e1071); library(caret); library(dplyr) })

results <- parLapply(cl, combo_kernel_pairs, process_combination)
stopCluster(cl)

# ----------------------------
# 8. Resultados
# ----------------------------
accuracies <- sapply(results, `[[`, "accuracy")
kernels_used <- sapply(results, `[[`, "kernel")
best_idx <- which.max(accuracies)
best <- results[[best_idx]]

cat("Melhor kernel:", best$kernel, "\n")
cat("Colunas removidas:", paste(best$cols, collapse = ", "), "\n")
cat("Acurácia:", round(best$accuracy, 2), "%\n")

# ----------------------------
# 9. Visualização dos resultados
# ----------------------------
df_acc <- data.frame(accuracy = accuracies, kernel = kernels_used)
ggplot(df_acc, aes(x = kernel, y = accuracy)) +
  geom_boxplot() +
  labs(title = "Distribuição das Acurácias por Kernel", x = "Kernel", y = "Acurácia (%)") +
  theme_minimal()
ggsave("plots/boxplot_acuracias.png")

df_scatter <- data.frame(num_cols_removed = sapply(results, function(x) length(x$cols)),
                         accuracy = accuracies, kernel = kernels_used)
ggplot(df_scatter, aes(x = num_cols_removed, y = accuracy, color = kernel)) +
  geom_point(alpha = 0.5) +
  labs(title = "Acurácia x Número de Colunas Removidas", x = "Colunas Removidas", y = "Acurácia (%)") +
  theme_minimal()
ggsave("plots/scatter_acuracias.png")

# Matriz de confusão
cm <- table(Predicted = best$pred, Actual = best$test$target)
ggplot(as.data.frame(cm), aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Matriz de Confusão", fill = "Frequência") +
  theme_minimal()
ggsave("plots/matriz_confusao.png")

# Curva ROC
roc_obj <- roc(as.numeric(best$test$target), as.numeric(best$pred))
png("plots/roc_curve.png")
plot(roc_obj, col = "blue", lwd = 2, main = "Curva ROC - Melhor Modelo")
dev.off()
cat("AUC:", round(auc(roc_obj), 4), "\n")
