#!/usr/bin/env python
# pricing_model.py
# Lógica principal de cálculo do pricing

import numpy as np
import time
import warnings
from scipy.optimize import fsolve, brentq
from src.config import *

def simulate_pricing(params):
    """
    Calcula o pricing ajustado ao risco e métricas para um contrato de empréstimo.
    
    Args:
        params (dict): Dicionário contendo PV, n, p_close_annual, p_rot_m, delay
        
    Returns:
        dict: Dicionário com as métricas de pricing calculadas
    """
    start = time.time()

    # --- 1) Extrai parâmetros básicos ------------------------------
    PV = float(params['PV'])
    n = int(params['n'])
    p_close_ann = float(params['p_close_annual'])
    p_rot_m = float(params['p_rot_m'])
    d = int(params['delay'])
    if n > 120:
        n = 120  # limite prático de parcelas

    # --- 2) Calcula hazards mensais -------------------------------
    # hazard de fechamento
    h_close = 1 - (1 - p_close_ann)**(1/12)
    # turnover inclui fechamento + rotatividade
    h_turnover = h_close + (1 - h_close) * (p_rot_m * 0.66)
    # default = parte do turnover que vira inadimplência
    h_default = h_turnover * P_INFORMALIDADE
    # delay = parte do turnover que só atrasa pagamentos
    h_delay = h_turnover * (1 - P_INFORMALIDADE)

    # --- 3) Constrói sobrevivência e densidades ------------------
    S = np.empty(n+1); S[0] = 1.0
    f_def = np.zeros(n+1); f_del = np.zeros(n+1)
    for u in range(1, n+1):
        S[u] = S[u-1] * (1 - h_default - h_delay)
        f_def[u] = S[u-1] * h_default
        f_del[u] = S[u-1] * h_delay
    S_n = S[n]  # probabilidade de chegar ao fim sem evento

    # --- 4) PMT livre de risco (anuidade clássica) ---------------
    PMT_base = PV * r_tar_m / (1 - (1 + r_tar_m)**(-n))

    # --- 5) EPV components e root‐finding para PMT_risco ---------
    def epv_components(PMT):
        # saldo devedor ao longo do tempo
        bal = np.empty(n+1); bal[0] = PV
        for t in range(1, n+1):
            juros = r_tar_m * bal[t-1]
            amort = PMT - juros
            bal[t] = bal[t-1] - amort

        # EPV sobrevivência
        surv_pv = S[n] * sum(PMT / (1 + r_tar_m)**t for t in range(1, n+1))
        # EPV atraso (pagamentos remarcados em t+d)
        delay_pv = sum(
            f_del[u] * (
                sum(PMT / (1 + r_tar_m)**t for t in range(1, u)) +
                sum(PMT / (1 + r_tar_m)**(t + d) for t in range(u, n+1))
            )
            for u in range(1, n+1)
        )
        # EPV default (perda parcial de saldo)
        def_pv = sum(
            f_def[u] * (
                sum(PMT / (1 + r_tar_m)**t for t in range(1, u)) +
                (1 - LGD) * bal[u-1] / (1 + r_tar_m)**u
            )
            for u in range(1, n+1)
        )
        return surv_pv, delay_pv, def_pv

    def epv_minus_pv(PMT):
        surv_pv, delay_pv, def_pv = epv_components(PMT)
        return (surv_pv + delay_pv + def_pv) - PV

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        PMT_risco = fsolve(epv_minus_pv, PMT_base)[0]
    surv_pv, delay_pv, def_pv = epv_components(PMT_risco)

    # --- 6) Spreads e TIR do cliente ----------------------------
    spread_valor = PMT_risco - PMT_base

    def f_client(r):
        return sum(PMT_risco / (1 + r)**t for t in range(1, n+1)) - PV
    try:
        r_min_m = brentq(f_client, -0.99, 1.0)
    except ValueError:
        r_min_m = 0.0
    R_min_a = (1 + r_min_m)**12 - 1
    spread = r_min_m - r_tar_m
    spread_a = R_min_a - R_BASE

    # --- 7) KPIs adicionais -------------------------------------
    P_default_total = f_def.sum()
    P_delay_total = f_del.sum()
    Expected_payments = S[:-1].sum()  # parcelas pagas **no prazo**
    Mean_time_to_event = sum(u * (f_def[u] + f_del[u]) for u in range(1, n+1))

    # **E(Duration)**: duração esperada do contrato
    E_Duration = (
        sum(u * f_def[u] for u in range(1, n+1)) +
        sum((n + d) * f_del[u] for u in range(1, n+1)) +
        n * S_n
    )

    # LGD ponderado
    bal = np.empty(n+1); bal[0] = PV
    for t in range(1, n+1):
        juros = r_tar_m * bal[t-1]
        amort = PMT_base - juros
        bal[t] = bal[t-1] - amort
    LGD_ponderado = (f_def[1:] * (LGD * bal[:-1])).sum() / PV

    elapsed = time.time() - start

    return {
        'PV': PV, 'n': n,
        'CDI_anual': CDI_ANUAL, 'Funding': FUNDING,
        'Custo_Operacional': CUSTO_OPERACIONAL, 'Margem': MARGEM,
        'R_base_anual': R_BASE,
        'h_close': h_close, 'h_turnover': h_turnover,
        'h_default': h_default, 'h_delay': h_delay,
        'S_n': S_n,
        'P_default_total': P_default_total, 'P_delay_total': P_delay_total,
        'EPV_surv': surv_pv, 'EPV_delay': delay_pv, 'EPV_default': def_pv,
        'Expected_payments': Expected_payments,
        'Mean_time_to_event': Mean_time_to_event,
        'E_Duration': E_Duration,
        'LGD_ponderado': LGD_ponderado,
        'PMT_base': PMT_base, 'PMT_risco': PMT_risco,
        'spread_valor': spread_valor,
        'r_tar_m': r_tar_m, 'r_min_m': r_min_m, 'spread': spread,
        'R_min_anual': R_min_a, 'spread_anual': spread_a,
        'calc_time_s': elapsed
    } 