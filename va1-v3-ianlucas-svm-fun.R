# ====================================================
# Projeto 1VA - SVM para Evasão Estudantil em R - CPAD
# Autor: Ian Lucas Almeida
# Curso: Mestrado em Informática Aplicada - UFRPE/PPGIA - 2025
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

# Gráficos
url_csv <- "https://raw.githubusercontent.com/ianlucasalmeida/projeto_1va-SVM/main/dataset/Student_Dropout_rate.csv"

# ----------------------------
# 5. Pré-processamento
# ----------------------------
X <- dataset %>% select(where(is.numeric))
y <- dataset$target

preproc <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preproc, X)

X_scaled <- X_scaled %>%
  select(where(is.numeric)) %>%
  select(where(~ all(is.finite(.)) && sd(.) > 0))

dataset_scaled <- bind_cols(as_tibble(X_scaled), target = y)

# ----------------------------
# 6. SVM inicial
# ----------------------------
svm_initial <- svm(target ~ ., data = dataset_scaled, kernel = "linear", scale = FALSE)
sv <- as.matrix(svm_initial$SV)
coefs <- as.matrix(svm_initial$coefs)
importances <- colSums(abs(sv * matrix(coefs, nrow = nrow(sv), ncol = ncol(sv), byrow = TRUE)))
names(importances) <- colnames(X_scaled)
top_features <- names(sort(importances, decreasing = TRUE))[1:10]

# ----------------------------
# 7. Combinações e função paralela (corrigida)
# ----------------------------
combos <- unlist(lapply(1:min(3, length(top_features)), function(r) combn(top_features, r, simplify = FALSE)), recursive = FALSE)
kernels <- c("linear", "radial", "polynomial")
combo_kernel_list <- lapply(combos, function(cmb) {
  lapply(kernels, function(k) list(combo = cmb, kernel = k))
}) %>% unlist(recursive = FALSE)

# Função segura
process_combination <- function(args) {
  tryCatch({
    cols <- args$combo
    kernel <- args$kernel
    df <- dataset_scaled %>% select(-all_of(cols))
    X <- df %>% select(-target)
    y <- df$target
    
    # Divisão treino/teste
    set.seed(5)
    idx <- createDataPartition(y, p = 0.8, list = FALSE)
    train <- bind_cols(X[idx, ], target = y[idx])
    test <- bind_cols(X[-idx, ], target = y[-idx])
    
    # ⚡ Verificar se treino tem ambas as classes
    if (length(unique(train$target)) < 2) return(list(accuracy = NA))
    
    # ⚡ Ajustar gamma dinamicamente
    gamma_value <- 1 / ncol(X)
    
    model <- svm(target ~ ., data = train, kernel = kernel, cost = 1, gamma = gamma_value, scale = FALSE)
    pred <- predict(model, test %>% select(-target))
    
    acc <- mean(pred == test$target) * 100
    list(accuracy = acc, kernel = kernel, cols = cols, pred = pred, test = test)
  }, error = function(e) list(accuracy = NA))
}

# Executar paralelização
cl <- makeCluster(2)
registerDoParallel(cl)
clusterExport(cl, c("dataset_scaled", "process_combination", "createDataPartition", "svm", "predict"))
clusterEvalQ(cl, { library(e1071); library(caret); library(dplyr) })
results <- parLapply(cl, combo_kernel_list, process_combination)
stopCluster(cl)

# ----------------------------
# 8. Resultados e Visualizações
# ----------------------------
# Garantir apenas resultados válidos
results <- results[!sapply(results, function(x) is.na(x$accuracy))]
if (length(results) == 0) stop("Nenhum resultado válido gerado.")

accuracies <- sapply(results, `[[`, "accuracy")
kernels_used <- sapply(results, `[[`, "kernel")
best_idx <- which.max(accuracies)
best <- results[[best_idx]]

# Visualização
dir.create("plots", showWarnings = FALSE)

# Gráfico: Boxplot acurácia
ggplot(data.frame(accuracy = accuracies, kernel = kernels_used), aes(x = kernel, y = accuracy)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Distribuição da Acurácia por Kernel", x = "Tipo de Kernel", y = "Acurácia (%)") +
  theme_minimal()
ggsave("plots/boxplot_acuracias.png")

# Gráfico: Dispersão colunas removidas vs acurácia
ggplot(data.frame(
  num_cols_removed = sapply(results, function(x) length(x$cols)),
  accuracy = accuracies,
  kernel = kernels_used
), aes(x = num_cols_removed, y = accuracy, color = kernel)) +
  geom_point(size = 2) +
  labs(title = "Acurácia vs Número de Colunas Removidas", x = "Colunas Removidas", y = "Acurácia (%)") +
  theme_minimal()
ggsave("plots/scatter_acuracias.png")

# Gráfico: Matriz de confusão
cm <- table(Predicted = best$pred, Actual = best$test$target)
ggplot(as.data.frame(cm), aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1.5) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Matriz de Confusão", fill = "Frequência") +
  theme_minimal()
ggsave("plots/matriz_confusao.png")

# Gráfico: Curva ROC
roc_obj <- roc(as.numeric(best$test$target), as.numeric(best$pred))
png("plots/roc_curve.png")
plot(roc_obj, col = "blue", lwd = 2, main = "Curva ROC - Melhor Modelo SVM")
dev.off()
cat("AUC:", round(auc(roc_obj), 4), "\n")

# Gráfico de Pirâmide Etária por Status (Dropout vs Graduate)

dataset %>%
  mutate(age_group = cut(age_at_enrollment, breaks = seq(15, 65, 5))) %>%
  count(target, age_group) %>%
  ggplot(aes(x = age_group, y = ifelse(target == "Dropout", -n, n), fill = target)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_y_continuous(labels = abs) +
  labs(title = "Distribuição de Idade por Status (Pirâmide Etária)", x = "Idade", y = "Quantidade de Alunos") +
  theme_minimal()
ggsave("plots/piramide_idade.png", width = 8, height = 6)

#Gráfico de Proporção de Evasão por Bolsa de Estudos

ggplot(dataset, aes(x = scholarship_holder, fill = target)) +
  geom_bar(position = "fill") +
  labs(title = "Proporção de Evasão por Bolsa de Estudos", x = "Possui Bolsa de Estudos?", y = "Proporção") +
  scale_fill_manual(values = c("#FF9999", "#99CCFF")) +
  theme_minimal()
ggsave("plots/evasao_bolsa_estudos.png", width = 8, height = 5)

# Boxplot: Nota de Admissão Separada por Status

ggplot(dataset, aes(x = target, y = admission_grade, fill = target)) +
  geom_boxplot() +
  labs(title = "Nota de Admissão por Status", x = "Status", y = "Nota de Admissão") +
  theme_minimal()
ggsave("plots/boxplot_nota_admissao.png", width = 8, height = 5)

# Histograma Comparativo de Número de Unidades Curriculares Completas

ggplot(dataset, aes(x = curricular_units_1st_sem_approved, fill = target)) +
  geom_histogram(position = "dodge", binwidth = 1) +
  labs(title = "Número de Disciplinas Aprovadas no 1º Semestre", x = "Disciplinas Aprovadas", y = "Quantidade de Estudantes") +
  theme_minimal()
ggsave("plots/disciplinas_aprovadas_1sem.png", width = 8, height = 5)

# Barplot: Evasão por Situação de Estudante (Part-time ou Full-time)

dataset <- dataset %>%
  mutate(situacao_estudante = ifelse(curricular_units_1st_sem_enrolled > 5, "Tempo Integral", "Tempo Parcial"))

ggplot(dataset, aes(x = situacao_estudante, fill = target)) +
  geom_bar(position = "fill") +
  labs(title = "Evasão por Situação de Estudo", x = "Situação", y = "Proporção") +
  theme_minimal()
ggsave("plots/evasao_situacao_estudo.png", width = 8, height = 5)

# Gráfico de Barras: Percentual de Evadidos vs Graduados em Relação a Bolsistas

dataset %>%
  group_by(scholarship_holder, target) %>%
  summarise(n = n()) %>%
  mutate(percent = n / sum(n) * 100) %>%
  ggplot(aes(x = scholarship_holder, y = percent, fill = target)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Percentual de Evasão entre Bolsistas e Não Bolsistas", x = "Possui Bolsa", y = "Percentual (%)") +
  theme_minimal()
ggsave("plots/evasao_percentual_bolsa.png", width = 8, height = 5)

# Scatter Plot: Relação entre Idade e Nota de Admissão

ggplot(dataset, aes(x = age_at_enrollment, y = admission_grade, color = target)) +
  geom_point(alpha = 0.5) +
  labs(title = "Idade vs Nota de Admissão por Status", x = "Idade", y = "Nota de Admissão") +
  theme_minimal()
ggsave("plots/idade_vs_nota_admissao.png", width = 8, height = 5)


