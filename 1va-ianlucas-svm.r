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
required_packages <- c("tidyverse", "e1071", "caret", "corrplot", 
                       "pROC", "janitor", "foreach", "doParallel")
walk(required_packages, ~ if (!require(.x, character.only = TRUE)) install.packages(.x))
lapply(required_packages, library, character.only = TRUE)

# ----------------------------
# 2. Funções auxiliares
# ----------------------------
# Função para calcular a moda
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# ----------------------------
# 3. Carregamento do dataset via GitHub
# ----------------------------
url_csv <- "https://raw.githubusercontent.com/ianlucasalmeida/projeto_1va-SVM/refs/heads/main/dataset/Student_Dropout_rate.csv"

dataset <- read_csv2(url_csv) %>% clean_names()

# Conversão e filtragem do target
dataset <- dataset %>%
  mutate(target = recode(target, "Graduate" = 0, "Dropout" = 1, "Enrolled" = 2)) %>%
  filter(target != 2)

# ----------------------------
# 4. Análise Exploratória
# ----------------------------
cat("Total de registros:", nrow(dataset), "\n")
cat("Taxa de evasão:", round(mean(dataset$target) * 100, 2), "%\n")

# Gráfico de idade por status
ggplot(dataset, aes(x = age_at_enrollment, fill = factor(target))) +
  geom_histogram(binwidth = 2) +
  labs(title = "Distribuição de Idade por Status", x = "Idade", y = "Contagem") +
  theme_minimal()
ggsave("imagens/idade_distribuicao.png", width = 8, height = 5)

# Moda da idade dos evasores
dropout_ages <- dataset %>% filter(target == 1) %>% pull(age_at_enrollment)
cat("Idade mais comum entre evasores:", Mode(dropout_ages), "anos\n")

# Correlação
cor_mat <- cor(dataset %>% select(where(is.numeric)), use = "complete.obs")
png("imagens/heatmap_correlacoes.png", width = 800, height = 600)
corrplot(cor_mat, method = "color", type = "upper", tl.col = "black", tl.srt = 45)
dev.off()

# ----------------------------
# 5. Pré-processamento
# ----------------------------
X <- dataset %>% select(-target)
y <- factor(dataset$target)
preproc <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preproc, X)
dataset_scaled <- bind_cols(as_tibble(X_scaled), target = y)

# ----------------------------
# 6. Modelo base com SVM (linear)
# ----------------------------
set.seed(42)
index <- createDataPartition(dataset_scaled$target, p = 0.8, list = FALSE)
train <- dataset_scaled[index, ]
test <- dataset_scaled[-index, ]

svm_model <- svm(target ~ ., data = train, kernel = "linear", probability = TRUE)
y_pred <- predict(svm_model, test %>% select(-target))
acc <- mean(y_pred == test$target) * 100
cat("Acurácia com kernel linear:", round(acc, 2), "%\n")

# ----------------------------
# 7. Matriz de confusão e curva ROC
# ----------------------------
cm <- table(Predito = y_pred, Real = test$target)
ggplot(as.data.frame(cm), aes(x = Predito, y = Real, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Matriz de Confusão", fill = "Frequência") +
  theme_minimal()
ggsave("imagens/matriz_confusao.png", width = 6, height = 5)

roc_obj <- roc(as.numeric(test$target), as.numeric(y_pred))
cat("AUC:", round(auc(roc_obj), 4), "\n")
png("imagens/roc_curve.png")
plot(roc_obj, col = "darkblue", lwd = 2, main = "Curva ROC")
dev.off()

# ----------------------------
# 8. Conclusão
# ----------------------------
cat("Modelo SVM implementado com sucesso.\n")
cat("Pronto para ser apresentado com base nos critérios do projeto da 1VA.\n")
