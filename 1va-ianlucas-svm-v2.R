# ====================================================
# Projeto 1VA - Aprendizado de Máquina com SVM em R
# Autor: Ian Lucas Almeida
# Curso: Mestrado em Informática Aplicada - UFRPE/PPGIA - 2025 | CPAD
# Data: Abril de 2025
# Objetivo: Prever evasão estudantil com SVM
# ====================================================

# ----------------------------
# 1. Instalar e carregar pacotes
# ----------------------------
required_packages <- c("dplyr", "ggplot2", "readr", "tibble", "e1071", "caret", 
                       "corrplot", "parallel", "doParallel", "pROC", "janitor")

invisible(lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg, dependencies = TRUE)
  library(pkg, character.only = TRUE)
}))

# ----------------------------
# 2. Função auxiliar
# ----------------------------
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# ----------------------------
# 3. Carregamento e limpeza
# ----------------------------
url_csv <- "https://raw.githubusercontent.com/ianlucasalmeida/projeto_1va-SVM/main/dataset/Student_Dropout_rate.csv"

dataset <- read_csv2(url_csv) %>%
  clean_names() %>%
  filter(target %in% c("Dropout", "Graduate")) %>%
  mutate(
    target = factor(target),
    gender = factor(gender),
    debtor = factor(debtor),
    scholarship_holder = factor(scholarship_holder),
    international = factor(international)
  )

# ----------------------------
# 4. Análise Exploratória e Gráficos
# ----------------------------
if (!dir.exists("plots")) dir.create("plots")

cat("Total de amostras:", nrow(dataset), "\n")
cat("Taxa de evasão:", round(mean(dataset$target == "Dropout") * 100, 2), "%\n")

# Gráfico 1: Distribuição de idade
ggplot(dataset, aes(x = age_at_enrollment, fill = target)) +
  geom_histogram(binwidth = 2, position = "stack") +
  labs(title = "Distribuição de Idade por Status", x = "Idade", y = "Contagem") +
  theme_minimal()
ggsave("plots/idade_distribuicao.png", width = 8, height = 5)

# Gráfico 2: Evasão por Gênero
ggplot(dataset, aes(x = gender, fill = target)) +
  geom_bar(position = "fill") +
  labs(title = "Evasão por Gênero", x = "Gênero", y = "Proporção") +
  theme_minimal()
ggsave("plots/evasao_genero.png", width = 8, height = 5)

# Gráfico 3: Notas 1º Semestre x Evasão
if ("curricular_units_1st_sem_grade" %in% names(dataset)) {
  ggplot(dataset, aes(x = curricular_units_1st_sem_grade, y = target)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "glm", method.args = list(family = "binomial")) +
    labs(title = "Nota do 1º Semestre x Evasão", x = "Nota", y = "Status") +
    theme_minimal()
  ggsave("plots/notas_evasao.png", width = 8, height = 5)
}

# Gráfico 4: Evasão por Estado Civil
ggplot(dataset, aes(x = factor(marital_status), fill = target)) +
  geom_bar(position = "fill") +
  labs(title = "Evasão por Estado Civil", x = "Estado Civil", y = "Proporção") +
  theme_minimal()
ggsave("plots/evasao_estado_civil.png", width = 8, height = 5)

# Gráfico 5: Evasão por Curso
ggplot(dataset, aes(x = factor(course), fill = target)) +
  geom_bar(position = "fill") +
  labs(title = "Evasão por Curso", x = "Curso", y = "Proporção") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("plots/evasao_por_curso.png", width = 10, height = 6)

# Gráfico 6: Nota de admissão x evasão
ggplot(dataset, aes(x = admission_grade, y = as.numeric(target))) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  labs(title = "Nota de Admissão x Evasão", x = "Nota", y = "Status") +
  theme_minimal()
ggsave("plots/nota_admissao_evasao.png", width = 8, height = 5)

# Gráfico 7: Taxa de desemprego
ggplot(dataset, aes(x = unemployment_rate, fill = target)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribuição da Taxa de Desemprego", x = "Taxa", y = "Densidade") +
  theme_minimal()
ggsave("plots/desemprego_status.png", width = 8, height = 5)

# Gráfico 8: Débito financeiro
ggplot(dataset, aes(x = debtor, fill = target)) +
  geom_bar(position = "fill") +
  labs(title = "Evasão por Débito", x = "Devedor", y = "Proporção") +
  theme_minimal()
ggsave("plots/evasao_devedor.png", width = 8, height = 5)

# Gráfico 9: Facetado - Gênero e Curso
ggplot(dataset, aes(x = gender, fill = target)) +
  geom_bar(position = "fill") +
  facet_wrap(~ course) +
  labs(title = "Evasão por Gênero em Cada Curso", x = "Gênero", y = "Proporção") +
  theme_minimal()
ggsave("plots/evasao_genero_curso_facetado.png", width = 10, height = 7)

# Gráfico 10: Heatmap de correlação (sem target)
dataset_numeric <- dataset %>% select(where(is.numeric))
png("plots/heatmap_correlacoes.png", width = 800, height = 600)
corrplot(cor(dataset_numeric, use = "complete.obs"), method = "color", type = "upper",
         tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.7)
dev.off()

# ----------------------------
# 5. Pré-processamento
# ----------------------------
X <- dataset_numeric
y <- dataset$target

preproc <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preproc, X)

X_scaled <- X_scaled %>%
  select(where(is.numeric)) %>%
  select(where(~ all(is.finite(.)) && sd(.) > 0))

dataset_scaled <- bind_cols(as_tibble(X_scaled), target = y)

# ----------------------------
# 6. SVM e seleção de variáveis
# ----------------------------
svm_initial <- svm(target ~ ., data = dataset_scaled, kernel = "linear")
sv <- as.matrix(svm_initial$SV)
coefs <- as.matrix(svm_initial$coefs)
importances <- colSums(abs(sv * matrix(coefs, nrow = nrow(sv), ncol = ncol(sv), byrow = TRUE)))
names(importances) <- colnames(X_scaled)
top_features <- names(sort(importances, decreasing = TRUE))[1:10]

# ----------------------------
# 7. Combinações e função paralela
# ----------------------------
combos <- unlist(lapply(1:min(3, length(top_features)), function(r) combn(top_features, r, simplify = FALSE)), recursive = FALSE)
kernels <- c("linear", "radial", "polynomial")
combo_kernel_list <- lapply(combos, function(cmb) {
  lapply(kernels, function(k) list(combo = cmb, kernel = k))
}) %>% unlist(recursive = FALSE)

process_combination <- function(args) {
  tryCatch({
    cols <- args$combo
    kernel <- args$kernel
    df <- dataset_scaled %>% select(-all_of(cols))
    X <- df %>% select(-target)
    y <- df$target
    idx <- createDataPartition(y, p = 0.8, list = FALSE)
    train <- bind_cols(X[idx, ], target = y[idx])
    test <- bind_cols(X[-idx, ], target = y[-idx])
    model <- svm(target ~ ., data = train, kernel = kernel, gamma = "auto")
    pred <- predict(model, test %>% select(-target))
    acc <- mean(pred == test$target) * 100
    list(accuracy = acc, kernel = kernel, cols = cols, pred = pred, test = test)
  }, error = function(e) list(accuracy = NA))
}

cl <- makeCluster(2)
registerDoParallel(cl)
clusterExport(cl, c("dataset_scaled", "process_combination", "createDataPartition", "svm", "predict"))
clusterEvalQ(cl, { library(e1071); library(caret); library(dplyr) })
results <- parLapply(cl, combo_kernel_list, process_combination)
stopCluster(cl)

# Garantir que há resultados válidos
if (length(results) == 0) {
  stop("❌ Nenhum resultado de combinação foi bem-sucedido. Verifique os dados ou o SVM.")
}

accuracies <- sapply(results, `[[`, "accuracy")
kernels_used <- sapply(results, `[[`, "kernel")
best_idx <- which.max(accuracies)

# Verificação extra para segurança
if (length(best_idx) == 0 || is.na(best_idx)) {
  stop("❌ Não foi possível encontrar o melhor índice.")
}

best <- results[[best_idx]]

accuracies <- sapply(results, `[[`, "accuracy")
kernels_used <- sapply(results, `[[`, "kernel")
best_idx <- which.max(accuracies)
best <- results[[best_idx]]

# ----------------------------
# 8. Visualizações do Modelo
# ----------------------------
# Gráfico 11: Boxplot das acurácias por kernel
ggplot(data.frame(accuracy = accuracies, kernel = kernels_used), aes(x = kernel, y = accuracy)) +
  geom_boxplot() +
  labs(title = "Acurácia por Kernel", x = "Kernel", y = "Acurácia (%)") +
  theme_minimal()
ggsave("plots/boxplot_acuracias.png")

# Gráfico 12: Dispersão de colunas removidas vs acurácia
ggplot(data.frame(
  num_cols_removed = sapply(results, function(x) length(x$cols)),
  accuracy = accuracies,
  kernel = kernels_used
), aes(x = num_cols_removed, y = accuracy, color = kernel)) +
  geom_point() +
  labs(title = "Acurácia vs Colunas Removidas", x = "Colunas Removidas", y = "Acurácia (%)") +
  theme_minimal()
ggsave("plots/scatter_acuracias.png")

# Gráfico 13: Matriz de confusão
cm <- table(Predicted = best$pred, Actual = best$test$target)
ggplot(as.data.frame(cm), aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Matriz de Confusão", fill = "Frequência") +
  theme_minimal()
ggsave("plots/matriz_confusao.png")

# Gráfico 14: Curva ROC
roc_obj <- roc(as.numeric(best$test$target), as.numeric(best$pred))
png("plots/roc_curve.png")
plot(roc_obj, col = "blue", lwd = 2, main = "Curva ROC - Melhor Modelo")
dev.off()
cat("📈 AUC:", round(auc(roc_obj), 4), "\n")
