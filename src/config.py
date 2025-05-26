#!/usr/bin/env python
# config.py
# Configurações e constantes para o modelo de pricing

# --- Versão da calculadora ----------------------------------------------
VERSION = "1.0.0"  # Versão da calculadora

# --- Configurações financeiras --------------------------------------------
CDI_ANUAL = 0.14    # 14% a.a.
FUNDING = CDI_ANUAL + 0.035  # CDI + 3,5%
CUSTO_OPERACIONAL = 0.02    # 2% a.a.
MARGEM = 0.04    # 4% a.a.

R_BASE = FUNDING + CUSTO_OPERACIONAL + MARGEM
r_tar_m = (1 + R_BASE)**(1/12) - 1  # taxa-meta mensal

# --- Configurações de risco ----------------------------------------------
P_INFORMALIDADE = 0.10    # 10% migração para informalidade
LGD = 0.80    # 80% perda em default estrutural
LGD_DEATH = 0.60    # 60% perda em caso de óbito

# --- Paths dos arquivos --------------------------------------------------
CSV_ROTATIVIDADE = 'dados/prob-rotatividade.csv'
CSV_FECHAMENTO = 'dados/new-risco-fechamento-porte.csv'
CSV_BASE = 'dados/base-producao_20250526.csv'
CSV_OBITO = 'dados/prob-anual-obito.csv'
OUTPUT_DIR = 'resultados'
OUTPUT_BASE = 'resultado_pricingv1'