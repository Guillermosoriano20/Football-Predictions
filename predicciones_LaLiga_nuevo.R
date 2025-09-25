# ============================================
# PREDICCIONES LALIGA — PIPELINE COMPLETO (NA-safe + RF clasif alineado)
# ============================================

# --- 0. Librerías ---
library(data.table)
library(dplyr)
library(MASS)           # glm.nb
library(randomForest)
library(xgboost)
library(fastDummies)
library(Metrics)

set.seed(123)

# --- 1. Cargar datos (ya con estilos) ---
datos <- fread("datos_LaLiga_estilos.csv")
datos$Match_Date <- as.Date(datos$Match_Date)

# --- 2. Añadir 'concedidos' por partido a partir del rival (crudo por partido) ---
datos[, partido_id := rep(1:(.N / 2), each = 2)]
if ("Rival" %in% names(datos)) setnames(datos, "Rival", "Rival_estilos")

vars_concedidas_raw <- c("Gls", "Sh", "SoT", "CrdY", "CrdR")
df_rival <- datos[, c("partido_id", "Team", vars_concedidas_raw), with = FALSE]
setnames(df_rival, old = c("Team", vars_concedidas_raw),
         new = c("Rival", paste0(vars_concedidas_raw, "_concedidos")))
datos <- merge(datos, df_rival, by = "partido_id", all.x = TRUE, allow.cartesian = TRUE)
datos <- datos[Team != Rival]

# --- 3. Construir medias móviles históricas (_L5) SIN FUGA (con shrinkage, NA-safe) ---
setDT(datos)
setorder(datos, Match_Date)

N <- 2   # ventana para históricos
k <- N

# a) Concedidas  ---- (NA-safe)
vars_concedidas <- c("Gls_concedidos","Sh_concedidos","SoT_concedidos","CrdY_concedidos","CrdR_concedidos")
datos[, games_prev := seq_len(.N) - 1L, by = .(Team, Home_Away)]
for (v in vars_concedidas) {
  prev_col <- paste0(v, "_prev"); exp_col <- paste0(v, "_exp")
  roll_col <- paste0(v, "_roll"); L_col  <- paste0(v, "_L", N)
  datos[, (prev_col) := shift(get(v)), by = .(Team, Home_Away)]
  datos[, (exp_col)  := cummean(get(prev_col)), by = .(Team, Home_Away)]
  datos[, (roll_col) := frollmean(get(prev_col), n = N), by = .(Team, Home_Away)]
  liga_mean <- mean(datos[[v]], na.rm = TRUE)
  datos[, w := pmin(games_prev / N, 1)]
  # NA-safe en los primeros partidos
  datos[, (L_col) := fifelse(
    games_prev < N,
    w * fifelse(is.na(get(exp_col)), liga_mean, get(exp_col)) + (1 - w) * liga_mean,
    get(roll_col)
  )]
  datos[, c(prev_col, exp_col, roll_col) := NULL]
}
datos[, c("w","games_prev") := NULL]

# b) Propias  ---- (NA-safe)
vars_propias <- c(
  "Gls","Sh","SoT","xG_Expected","npxG_Expected","xAG_Expected","SCA_SCA","GCA_SCA",
  "Cmp_Passes","Att_Passes","Cmp_percent_Passes","PrgP_Passes","Carries_Carries","PrgC_Carries",
  "Att_Take_Ons","Succ_Take_Ons","Touches","Int","Blocks","tackles","centros","faltas","paradas",
  "recuperaciones","Ast","PK","PKatt","CrdY","CrdR"
)
datos[, games_prev2 := seq_len(.N) - 1L, by = .(Team, Home_Away)]
for (v in vars_propias) {
  prev_col <- paste0(v, "_prev"); exp_col <- paste0(v, "_exp")
  roll_col <- paste0(v, "_roll"); L_col  <- paste0(v, "_L", N)
  datos[, (prev_col) := shift(get(v)), by = .(Team, Home_Away)]
  datos[, (exp_col)  := cummean(get(prev_col)), by = .(Team, Home_Away)]
  datos[, (roll_col) := frollmean(get(prev_col), n = N), by = .(Team, Home_Away)]
  liga_mean <- mean(datos[[v]], na.rm = TRUE)
  datos[, w2 := pmin(games_prev2 / N, 1)]
  datos[, (L_col) := fifelse(
    games_prev2 < N,
    w2 * fifelse(is.na(get(exp_col)), liga_mean, get(exp_col)) + (1 - w2) * liga_mean,
    get(roll_col)
  )]
  datos[, c(prev_col, exp_col, roll_col) := NULL]
}
datos[, c("w2","games_prev2") := NULL]

# --- 4. Dataset a nivel partido: SOLO _L5 + interacciones ---
locales    <- datos[seq(1, .N, by = 2)]
visitantes <- datos[seq(2, .N, by = 2)]

variables_base <- c(
  paste0(vars_propias, "_L", N),
  paste0(c("Gls_concedidos","Sh_concedidos","SoT_concedidos","CrdY_concedidos","CrdR_concedidos"), "_L", N)
)

# Matriz partido (predictores _L5)
X <- data.frame(partido_id = locales$partido_id,
                lapply(variables_base, function(v) locales[[v]]))
colnames(X)[-1] <- paste0("L_", variables_base)
Y <- data.frame(lapply(variables_base, function(v) visitantes[[v]]))
colnames(Y) <- paste0("V_", variables_base)
datos_partidos <- bind_cols(X, Y)

# Añadir TARGETS reales (valores del partido)
datos_partidos$L_Gls  <- locales$Gls
datos_partidos$V_Gls  <- visitantes$Gls
datos_partidos$L_Sh   <- locales$Sh
datos_partidos$V_Sh   <- visitantes$Sh
datos_partidos$L_CrdY <- locales$CrdY
datos_partidos$V_CrdY <- visitantes$CrdY

# Interacciones ataque × defensa (pares clave)
pares <- list(
  c(paste0("Sh_L",N),          paste0("Sh_concedidos_L",N)),
  c(paste0("SoT_L",N),         paste0("SoT_concedidos_L",N)),
  c(paste0("xG_Expected_L",N), paste0("SoT_concedidos_L",N)),
  c(paste0("Gls_L",N),         paste0("Gls_concedidos_L",N))
)
for (p in pares) {
  att <- p[1]; def <- p[2]
  a1 <- paste0("L_", att); d1 <- paste0("V_", def)
  a2 <- paste0("V_", att); d2 <- paste0("L_", def)
  datos_partidos[[paste0(a1, "_x_", d1)]]     <- datos_partidos[[a1]] * datos_partidos[[d1]]
  datos_partidos[[paste0(a1, "_ratio_", d1)]] <- datos_partidos[[a1]] / pmax(datos_partidos[[d1]], 1e-6)
  datos_partidos[[paste0(a1, "_diff_", d1)]]  <- datos_partidos[[a1]] - datos_partidos[[d1]]
  datos_partidos[[paste0(a2, "_x_", d2)]]     <- datos_partidos[[a2]] * datos_partidos[[d2]]
  datos_partidos[[paste0(a2, "_ratio_", d2)]] <- datos_partidos[[a2]] / pmax(datos_partidos[[d2]], 1e-6)
  datos_partidos[[paste0(a2, "_diff_", d2)]]  <- datos_partidos[[a2]] - datos_partidos[[d2]]
}

# --- 5. Añadir estilos + dummies ---
estilos_cols <- c("Estilo_Ofensivo", "Estilo_Defensivo", "Estilo_Global",
                  "Estilo_Ofensivo_Rival", "Estilo_Defensivo_Rival", "Estilo_Global_Rival")
estilos_local <- locales[, c("partido_id", estilos_cols), with = FALSE]
setnames(estilos_local, estilos_cols, paste0("L_", estilos_cols))
estilos_visitante <- visitantes[, c("partido_id", estilos_cols), with = FALSE]
setnames(estilos_visitante, estilos_cols, paste0("V_", estilos_cols))
datos_partidos <- bind_cols(datos_partidos, estilos_local[, -1], estilos_visitante[, -1])

datos_partidos <- dummy_cols(
  datos_partidos,
  select_columns = c("L_Estilo_Ofensivo", "L_Estilo_Defensivo", "L_Estilo_Global",
                     "V_Estilo_Ofensivo", "V_Estilo_Defensivo", "V_Estilo_Global",
                     "L_Estilo_Ofensivo_Rival", "L_Estilo_Defensivo_Rival", "L_Estilo_Global_Rival",
                     "V_Estilo_Ofensivo_Rival", "V_Estilo_Defensivo_Rival", "V_Estilo_Global_Rival"),
  remove_first_dummy = TRUE, remove_selected_columns = TRUE
)

# --- 6. Targets y selección de predictores por objetivo ---
targets <- c("L_Gls", "V_Gls", "L_Sh", "V_Sh", "L_CrdY", "V_CrdY")

preds_por_target <- list(
  L_Gls = grep(paste0("^L_(Gls|Sh|SoT|xG_Expected)_L", N, "$|^V_(Gls|Sh|SoT)_concedidos_L", N, "$|_x_|_ratio_|_diff_|^L_Estilo_|^V_Estilo_"),
               names(datos_partidos), value = TRUE),
  V_Gls = grep(paste0("^V_(Gls|Sh|SoT|xG_Expected)_L", N, "$|^L_(Gls|Sh|SoT)_concedidos_L", N, "$|_x_|_ratio_|_diff_|^L_Estilo_|^V_Estilo_"),
               names(datos_partidos), value = TRUE),
  L_Sh  = grep(paste0("^L_(Sh|xG_Expected|Att_Take_Ons|centros)_L", N, "$|^V_Sh_concedidos_L", N, "$|_x_|_ratio_|_diff_|^L_Estilo_|^V_Estilo_"),
               names(datos_partidos), value = TRUE),
  V_Sh  = grep(paste0("^V_(Sh|xG_Expected|Att_Take_Ons|centros)_L", N, "$|^L_Sh_concedidos_L", N, "$|_x_|_ratio_|_diff_|^L_Estilo_|^V_Estilo_"),
               names(datos_partidos), value = TRUE),
  L_CrdY= grep(paste0("^L_(CrdY|faltas|tackles)_L", N, "$|^V_CrdY_concedidos_L", N, "$|^L_Estilo_|^V_Estilo_"),
               names(datos_partidos), value = TRUE),
  V_CrdY= grep(paste0("^V_(CrdY|faltas|tackles)_L", N, "$|^L_CrdY_concedidos_L", N, "$|^L_Estilo_|^V_Estilo_"),
               names(datos_partidos), value = TRUE)
)

# --- 7. Entrenar modelos de REGRESIÓN (robustos + imputador por target) ---
modelos_entrenados <- list()
for (var in targets) {
  y0 <- datos_partidos[[var]]
  x0 <- datos_partidos[, preds_por_target[[var]], drop = FALSE]
  
  # quitar columnas constantes
  const_cols <- names(x0)[sapply(x0, function(z) {
    zf <- z[is.finite(z)]
    length(unique(zf)) <= 1
  })]
  if (length(const_cols)) x0 <- x0[, setdiff(names(x0), const_cols), drop = FALSE]
  
  # df y limpieza de NA/Inf
  df <- data.frame(y = y0, x0, check.names = FALSE)
  for (nm in names(df)) if (is.numeric(df[[nm]]) && any(is.infinite(df[[nm]]))) df[[nm]][is.infinite(df[[nm]])] <- NA
  df <- df[stats::complete.cases(df), , drop = FALSE]
  if (nrow(df) < 40 || length(unique(df$y)) < 2) {
    warning(sprintf("Saltando %s: pocos datos o sin variabilidad.", var))
    next
  }
  
  # GLM NB con fallback Poisson
  modelo_glm <- try(MASS::glm.nb(y ~ ., data = df, control = glm.control(maxit = 50)), silent = TRUE)
  if (inherits(modelo_glm, "try-error") || any(!is.finite(coef(modelo_glm)))) {
    modelo_glm <- glm(y ~ ., family = poisson(), data = df, control = glm.control(maxit = 50))
  }
  
  # RF
  modelo_rf <- randomForest(x = df[, -1, drop = FALSE], y = df$y, na.action = na.omit)
  
  # XGB
  x_num <- data.frame(lapply(df[, -1, drop = FALSE], function(col)
    if (is.character(col)) as.numeric(as.factor(col)) else col))
  dtrain <- xgb.DMatrix(data = as.matrix(x_num), label = df$y)
  modelo_xgb <- xgboost(data = dtrain, objective = "reg:squarederror",
                        nrounds = 200, max_depth = 4, eta = 0.1,
                        subsample = 0.8, colsample_bytree = 0.8, verbose = 0)
  
  # IMPUTADOR por variable: medianas del train
  medianas_train <- sapply(df[, -1, drop = FALSE], function(z) if (is.numeric(z)) stats::median(z, na.rm = TRUE) else NA_real_)
  
  modelos_entrenados[[var]] <- list(
    glm = modelo_glm, rf = modelo_rf, xgb = modelo_xgb,
    predictors = colnames(df)[-1],
    imputer = medianas_train
  )
}

# --- 8. Clasificación (RF + XGB) guardando columnas usadas ---
datos_partidos$Resultado <- with(datos_partidos, factor(
  ifelse(L_Gls > V_Gls, "Local", ifelse(L_Gls < V_Gls, "Visitante", "Empate")),
  levels = c("Local","Empate","Visitante")
))

# Usamos como features de clasificación la unión de predictores de regresión
features_union <- unique(unlist(preds_por_target))

x_clasif <- datos_partidos[, features_union, drop = FALSE]
y_clasif <- datos_partidos$Resultado

# Limpieza e imputación
for (nm in names(x_clasif)) if (is.numeric(x_clasif[[nm]]) && any(is.infinite(x_clasif[[nm]]))) x_clasif[[nm]][is.infinite(x_clasif[[nm]])] <- NA
x_clasif <- randomForest::na.roughfix(x_clasif)

# RF
modelo_rf_clasif <- randomForest(x = x_clasif, y = y_clasif)
cols_rf_clasif <- colnames(x_clasif)  # columnas exactas usadas en RF
# Imputador de clasificación RF: medianas por columna (sobre x_clasif ya imputado)
imputer_rf <- sapply(x_clasif, function(z) {
  if (is.numeric(z)) stats::median(z, na.rm = TRUE) else NA_real_
})

# XGB
x_clasif_numeric <- data.frame(lapply(x_clasif, function(col) if (is.character(col)) as.numeric(as.factor(col)) else col))
dtrain_clasif <- xgb.DMatrix(data = as.matrix(x_clasif_numeric), label = as.numeric(y_clasif) - 1)
modelo_xgb_clasif <- xgboost(data = dtrain_clasif, objective = "multi:softprob", num_class = 3, nrounds = 200,
                             max_depth = 4, eta = 0.1, subsample = 0.8, colsample_bytree = 0.8, verbose = 0)
cols_xgb_clasif <- colnames(x_clasif_numeric)

# === Guardar TODO en la lista ===
modelos_clasificacion <- list(
  rf = modelo_rf_clasif,
  xgb = modelo_xgb_clasif,
  niveles = c("Local","Empate","Visitante"),
  predictors_rf = cols_rf_clasif,
  predictors_xgb = cols_xgb_clasif,
  imputer_rf = imputer_rf   # <-- aquí guardas el imputador
)
# --- 9. Predicción de partidos nuevos (con imputación y alineación de columnas) ---
predecir_partidos_nuevos <- function(
    equipos_locales, equipos_visitantes,
    datos, modelos_entrenados, modelos_clasificacion,
    preds_por_target, targets, n_partidos = 5
) {
  stopifnot(length(equipos_locales) == length(equipos_visitantes))
  
  get_media <- function(equipo, tipo, var, n_partidos = 5, prior_mean = NULL, k = n_partidos) {
    df <- datos[datos$Team == equipo & datos$Home_Away == tipo, ]
    df <- df[order(df$Match_Date, decreasing = TRUE), ]
    # si ya es _L5, usamos el último valor (ya histórico)
    if (grepl("_L\\d+$", var)) return(df[[var]][1])
    vals <- head(df[[var]], n_partidos)
    g <- sum(!is.na(vals))
    if (is.null(prior_mean)) prior_mean <- mean(datos[[var]], na.rm = TRUE)
    if (g == 0) return(prior_mean)
    m_team <- mean(vals, na.rm = TRUE)
    (g/(g + k)) * m_team + (k/(g + k)) * prior_mean
  }
  get_estilo <- function(equipo, tipo, var) {
    df <- datos[datos$Team == equipo & datos$Home_Away == tipo, ]
    df <- df[order(df$Match_Date, decreasing = TRUE), ]
    df[[var]][1]
  }
  
  resultados <- list()
  features_union <- unique(unlist(preds_por_target))
  base_cols <- features_union[!grepl("(_x_|_ratio_|_diff_)", features_union)]  # sin interacciones (se recrean)
  
  for (i in seq_along(equipos_locales)) {
    local <- equipos_locales[i]; visitante <- equipos_visitantes[i]
    input <- data.frame(matrix(ncol = 0, nrow = 1))
    
    # 1) Medias recientes para columnas base (L_/V_ *_L5)
    for (v in base_cols) {
      if (startsWith(v, "L_")) input[[v]] <- get_media(local, "Home", sub("L_", "", v), n_partidos = n_partidos)
      else if (startsWith(v, "V_")) input[[v]] <- get_media(visitante, "Away", sub("V_", "", v), n_partidos = n_partidos)
    }
    
    # 2) Estilos recientes y dummies
    for (estilo in c("Estilo_Ofensivo","Estilo_Defensivo","Estilo_Global")) {
      input[[paste0("L_", estilo)]] <- get_estilo(local, "Home", estilo)
      input[[paste0("V_", estilo)]] <- get_estilo(visitante, "Away", estilo)
      input[[paste0("L_", estilo, "_Rival")]] <- get_estilo(visitante, "Away", estilo)
      input[[paste0("V_", estilo, "_Rival")]] <- get_estilo(local, "Home", estilo)
    }
    input <- dummy_cols(
      input,
      select_columns = c("L_Estilo_Ofensivo", "L_Estilo_Defensivo", "L_Estilo_Global",
                         "V_Estilo_Ofensivo", "V_Estilo_Defensivo", "V_Estilo_Global",
                         "L_Estilo_Ofensivo_Rival", "L_Estilo_Defensivo_Rival", "L_Estilo_Global_Rival",
                         "V_Estilo_Ofensivo_Rival", "V_Estilo_Defensivo_Rival", "V_Estilo_Global_Rival"),
      remove_selected_columns = TRUE, remove_first_dummy = TRUE
    )
    
    # 3) Reproducir interacciones del train
    for (p in list(
      c(paste0("Sh_L",N),          paste0("Sh_concedidos_L",N)),
      c(paste0("SoT_L",N),         paste0("SoT_concedidos_L",N)),
      c(paste0("xG_Expected_L",N), paste0("SoT_concedidos_L",N)),
      c(paste0("Gls_L",N),         paste0("Gls_concedidos_L",N))
    )) {
      att <- p[1]; def <- p[2]
      a1 <- paste0("L_", att); d1 <- paste0("V_", def)
      a2 <- paste0("V_", att); d2 <- paste0("L_", def)
      input[[paste0(a1, "_x_", d1)]]     <- input[[a1]] * input[[d1]]
      input[[paste0(a1, "_ratio_", d1)]] <- input[[a1]] / pmax(input[[d1]], 1e-6)
      input[[paste0(a1, "_diff_", d1)]]  <- input[[a1]] - input[[d1]]
      input[[paste0(a2, "_x_", d2)]]     <- input[[a2]] * input[[d2]]
      input[[paste0(a2, "_ratio_", d2)]] <- input[[a2]] / pmax(input[[d2]], 1e-6)
      input[[paste0(a2, "_diff_", d2)]]  <- input[[a2]] - input[[d2]]
    }
    
    # 4) Completar faltantes del universo de features (union de regresión)
    faltan <- setdiff(features_union, names(input))
    for (col in faltan) input[[col]] <- 0
    # limpiar Inf -> NA
    for (nm in names(input)) if (is.numeric(input[[nm]]) && any(is.infinite(input[[nm]]))) input[[nm]][is.infinite(input[[nm]])] <- NA
    
    # ===========================
    # PREDICCIONES DE REGRESIÓN
    # ===========================
    fila <- data.frame(Equipo_Local = local, Equipo_Visitante = visitante)
    
    for (var in targets) {
      if (is.null(modelos_entrenados[[var]])) next
      pred_cols <- modelos_entrenados[[var]]$predictors
      imp <- modelos_entrenados[[var]]$imputer
      
      input_sub <- input[, pred_cols, drop = FALSE]
      
      # Imputación por mediana del train (col a col)
      for (nm in names(imp)) {
        if (!nm %in% names(input_sub)) input_sub[[nm]] <- imp[[nm]]
        if (is.numeric(input_sub[[nm]]) && is.na(input_sub[[nm]][1])) input_sub[[nm]][1] <- imp[[nm]]
      }
      if (anyNA(input_sub)) input_sub <- randomForest::na.roughfix(input_sub)
      
      # GLM
      pred_glm <- try(predict(modelos_entrenados[[var]]$glm, newdata = input_sub, type = "response"), silent = TRUE)
      if (inherits(pred_glm, "try-error") || !is.finite(pred_glm)) pred_glm <- NA_real_
      fila[[paste0(var, "_GLM")]] <- pred_glm
      
      # XGB (numérico)
      input_num_sub <- data.frame(lapply(input_sub, function(col) if (is.character(col)) as.numeric(as.factor(col)) else col))
      fila[[paste0(var, "_XGB")]] <- predict(modelos_entrenados[[var]]$xgb, newdata = as.matrix(input_num_sub))
      
      # RF + Intervalos
      fila[[paste0(var, "_RF")]] <- predict(modelos_entrenados[[var]]$rf, newdata = input_sub)
      pred_rf_all <- predict(modelos_entrenados[[var]]$rf, newdata = input_sub, predict.all = TRUE)
      rf_vals <- pred_rf_all$individual
      if (is.matrix(rf_vals)) {
        fila[[paste0(var, "_RF_inf")]] <- apply(rf_vals, 1, function(r) stats::quantile(r, 0.025))
        fila[[paste0(var, "_RF_sup")]] <- apply(rf_vals, 1, function(r) stats::quantile(r, 0.975))
      } else {
        fila[[paste0(var, "_RF_inf")]] <- stats::quantile(rf_vals, 0.025)
        fila[[paste0(var, "_RF_sup")]] <- stats::quantile(rf_vals, 0.975)
      }
    }
    
    # ===========================
    # PREDICCIONES DE CLASIFICACIÓN
    # ===========================
    niveles <- modelos_clasificacion$niveles
    
    # === CLASIFICACIÓN RF (NA-safe + columnas alineadas) ===
    
    cols_rf <- modelos_clasificacion$predictors_rf
    imp_rf  <- modelos_clasificacion$imputer_rf
    
    # Construimos EXACTAMENTE las columnas de entrenamiento, imputando donde falte
    input_rf <- data.frame(matrix(nrow = 1, ncol = 0))
    
    for (nm in cols_rf) {
      # valor presente en 'input' o NA si no existe
      val <- if (nm %in% names(input)) input[[nm]][1] else NA_real_
      
      # limpiar infinitos
      if (is.numeric(val) && is.infinite(val)) val <- NA_real_
      
      # imputar con la mediana del train; si no existe mediana, usar 0
      fill <- if (!is.null(imp_rf[[nm]]) && is.finite(imp_rf[[nm]])) imp_rf[[nm]] else 0
      if (is.na(val)) val <- fill
      
      # asegurar numérico
      input_rf[[nm]] <- as.numeric(val)
    }
    
    # ordenar columnas exactamente como en el entrenamiento
    input_rf <- input_rf[, cols_rf, drop = FALSE]
    
    # predecir probabilidades
    prob_rf <- predict(modelos_clasificacion$rf, newdata = input_rf, type = "prob")
    
    # --- XGB: usar las mismas columnas numéricas del train ---
    cols_xgb <- modelos_clasificacion$predictors_xgb
    input_xgb <- data.frame(lapply(input[, cols_rf, drop = FALSE], function(col) if (is.character(col)) as.numeric(as.factor(col)) else col))
    # garantizar mismas columnas/orden que en train-xgb
    faltan_xgb <- setdiff(cols_xgb, names(input_xgb))
    for (col in faltan_xgb) input_xgb[[col]] <- 0
    extra_xgb <- setdiff(names(input_xgb), cols_xgb)
    if (length(extra_xgb)) input_xgb <- input_xgb[, !(names(input_xgb) %in% extra_xgb), drop = FALSE]
    input_xgb <- as.matrix(input_xgb[, cols_xgb, drop = FALSE])
    
    prob_xgb <- predict(modelos_clasificacion$xgb, newdata = input_xgb)
    prob_xgb <- matrix(prob_xgb, ncol = length(niveles), byrow = TRUE)
    colnames(prob_xgb) <- niveles
    
    for (nivel in niveles) {
      fila[[paste0("P_RF_", nivel)]]  <- prob_rf[, nivel]
      fila[[paste0("P_XGB_", nivel)]] <- prob_xgb[, nivel]
    }
    
    resultados[[i]] <- fila
  }
  
  df <- bind_rows(resultados)
  write.csv(df, "predicciones_partidos_nuevos.csv", row.names = FALSE)
  return(df)
}

# --- 10. Ejemplo de uso ---
equipos_locales <- c("Espanyol","Athletic Club","Sevilla","Levante","Getafe","Real Sociedad","Atlético Madrid","Osasuna","Oviedo")
equipos_visitantes <- c("Valencia","Girona","Villarreal","Real Madrid","Alavés","Mallorca","Rayo Vallecano","Elche","Barcelona")

resultado_nuevo <- predecir_partidos_nuevos(
  equipos_locales = equipos_locales,
  equipos_visitantes = equipos_visitantes,
  datos = datos,
  modelos_entrenados = modelos_entrenados,
  modelos_clasificacion = modelos_clasificacion,
  preds_por_target = preds_por_target,
  targets = targets,
  n_partidos = N
)
write.csv(resultado_nuevo, "Predicciones.csv", row.names = FALSE)
print(head(resultado_nuevo))




library(dplyr)
library(MASS)
library(randomForest)
library(xgboost)

validacion_rolling_regresion <- function(datos_partidos, preds_por_target, targets, n_inicial = 200) {
  n_total <- nrow(datos_partidos)
  res_reg <- data.frame()
  
  for (i in seq(n_inicial, n_total - 1)) {
    train_set <- datos_partidos[1:i, ]
    test_set  <- datos_partidos[i+1, , drop = FALSE]
    
    for (var in targets) {
      x_tr <- train_set[, preds_por_target[[var]], drop = FALSE]
      y_tr <- train_set[[var]]
      
      # quitar columnas constantes
      const_cols <- names(x_tr)[sapply(x_tr, function(z) {
        zf <- z[is.finite(z)]
        length(unique(zf)) <= 1
      })]
      if (length(const_cols)) x_tr <- x_tr[, setdiff(names(x_tr), const_cols), drop = FALSE]
      
      # DF train y limpieza
      df_tr <- data.frame(y = y_tr, x_tr, check.names = FALSE)
      for (nm in names(df_tr)) {
        if (is.numeric(df_tr[[nm]]) && any(is.infinite(df_tr[[nm]]))) df_tr[[nm]][is.infinite(df_tr[[nm]])] <- NA
      }
      df_tr <- df_tr[stats::complete.cases(df_tr), , drop = FALSE]
      if (nrow(df_tr) < 40 || length(unique(df_tr$y)) < 2) next
      
      # Imputador por medianas del train (para usar en test)
      imputer <- sapply(df_tr[, -1, drop = FALSE], function(z) if (is.numeric(z)) stats::median(z, na.rm = TRUE) else NA_real_)
      
      # Modelos
      m_glm <- try(MASS::glm.nb(y ~ ., data = df_tr, control = glm.control(maxit = 50)), silent = TRUE)
      if (inherits(m_glm, "try-error") || any(!is.finite(coef(m_glm)))) {
        m_glm <- glm(y ~ ., family = poisson(), data = df_tr, control = glm.control(maxit = 50))
      }
      m_rf  <- randomForest(x = df_tr[, -1, drop = FALSE], y = df_tr$y, na.action = na.omit)
      
      xnum_tr <- data.frame(lapply(df_tr[, -1, drop = FALSE], function(col)
        if (is.character(col)) as.numeric(as.factor(col)) else col))
      dtr     <- xgb.DMatrix(as.matrix(xnum_tr), label = df_tr$y)
      m_xgb   <- xgboost(data = dtr, objective = "reg:squarederror",
                         nrounds = 200, max_depth = 4, eta = 0.1,
                         subsample = 0.8, colsample_bytree = 0.8, verbose = 0)
      
      # Test (1 fila) con imputación por medianas del train
      x_te <- test_set[, colnames(df_tr)[-1], drop = FALSE]  # mismas columnas que usó el train
      for (nm in names(x_te)) if (is.numeric(x_te[[nm]]) && any(is.infinite(x_te[[nm]]))) x_te[[nm]][is.infinite(x_te[[nm]])] <- NA
      for (nm in names(x_te)) if (is.na(x_te[[nm]][1])) x_te[[nm]][1] <- imputer[[nm]]
      if (anyNA(x_te)) x_te <- randomForest::na.roughfix(x_te)
      
      y_te <- as.numeric(test_set[[var]])
      
      # Predicciones
      p_glm <- try(predict(m_glm, newdata = x_te, type = "response"), silent = TRUE); if (inherits(p_glm, "try-error")) p_glm <- NA_real_
      p_rf  <- predict(m_rf,  newdata = x_te)
      p_xgb <- {
        x_te_num <- data.frame(lapply(x_te, function(col) if (is.character(col)) as.numeric(as.factor(col)) else col))
        predict(m_xgb, newdata = as.matrix(x_te_num))
      }
      
      # Errores absolutos
      res_reg <- rbind(res_reg,
                       data.frame(Partido = i+1, Variable = var, Modelo = "GLM", ErrorAbs = abs(y_te - p_glm)),
                       data.frame(Partido = i+1, Variable = var, Modelo = "RF",  ErrorAbs = abs(y_te - p_rf)),
                       data.frame(Partido = i+1, Variable = var, Modelo = "XGB", ErrorAbs = abs(y_te - p_xgb)))
    }
  }
  
  resumen_reg <- aggregate(ErrorAbs ~ Variable + Modelo, data = res_reg, FUN = function(z) mean(z, na.rm = TRUE))
  list(detalle_regresion = res_reg, resumen_regresion = resumen_reg)
}

# === Ejecutar rolling y escoger el mejor modelo por variable ===
resultados_rolling <- validacion_rolling_regresion(
  datos_partidos = datos_partidos,
  preds_por_target = preds_por_target,
  targets = targets,
  n_inicial = 200   # ajusta si quieres
)

mejor_por_var <- resultados_rolling$resumen_regresion %>%
  group_by(Variable) %>%
  slice_min(order_by = ErrorAbs, n = 1, with_ties = FALSE) %>%
  arrange(Variable)

cat("\n==== Mejor modelo por variable (MAE, rolling) ====\n")
print(mejor_por_var)
write.csv(mejor_por_var, "mejor_modelo_por_variable.csv", row.names = FALSE)


