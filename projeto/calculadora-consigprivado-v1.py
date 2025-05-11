#!/usr/bin/env python
# pricingv1.py
# Rodrigo Miranda – 2025-05-11
# ------------------------------------------------------------
"""
Pricing Loan Calculator v1
Objetivo:
    Calcular o preço (PMT_risco), juros mínimo sustentável e métricas chave de um empréstimo consignado privado,
    dado:
      - PV: valor presente do empréstimo
      - n: número de parcelas
      - probabilidades de fechamento anual e rotatividade mensal (hazards), delay de pagamento
      - LGD, risco de informalidade,
      - CDI, funding, custo operacional, margem
Fluxo geral:
  1) Carrega dados de riscos e base de contratos
  2) Para cada contrato:
     a) Converte probabilidades em hazards mensais:
        h_close, h_turnover, h_default, h_delay
     b) Constrói curva de sobrevivência S[u] e densidades f_def[u], f_del[u]
     c) Calcula PMT_base pela anuidade sem risco
     d) Ajusta PMT_risco via root-finding de EPV_total(PMT) = PV
     e) Calcula TIR do cliente r_min_m (root de NPV(PMT_risco) = PV)
     f) Deriva spreads:
        • spread_valor = PMT_risco – PMT_base
        • spread = r_min_m – r_tar_m
        • R_min_anual e spread_anual
     g) Calcula KPIs extras:
        • Expected_payments (parcelas pagas em prazo)
        • Mean_time_to_event (meses até primeiro evento)
        • E_Duration (duração esperada do contrato, incluindo atraso)
        • LGD_ponderado
  3) Exporta CSV com colunas na ordem solicitada.
"""
# Adaptado de calculadora-perdav8.py :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

import pandas as pd
import numpy as np
import argparse
import time
import warnings
from scipy.optimize import fsolve, brentq
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# --- Configurações iniciais --------------------------------------------
CDI_ANUAL         = 0.14    # 14% a.a.
FUNDING           = CDI_ANUAL + 0.035  # CDI + 3,5%
CUSTO_OPERACIONAL = 0.02    # 2% a.a.
MARGEM            = 0.04    # 4% a.a.

R_BASE            = FUNDING + CUSTO_OPERACIONAL + MARGEM
r_tar_m           = (1 + R_BASE)**(1/12) - 1  # taxa-meta mensal

P_INFORMALIDADE   = 0.10    # 10% migração para informalidade
LGD               = 0.80    # 80% perda em default estrutural

CSV_ROTATIVIDADE  = 'dados/prob-rotatividade.csv'
CSV_FECHAMENTO    = 'dados/risco-fechamento-porte.csv'
CSV_BASE          = 'dados/base-producao-neoway_20250503.csv'
OUTPUT_DEFAULT    = 'resultado_pricingv1.csv'

def get_risco_fechamento_annual(row, risco_df):
    """
    Busca a probabilidade anual de fechamento pelo CNAE, porte e idade da empresa.
    """
    sec   = row.get('cnae_section') or 'A'
    emp   = float(row['tempo_empresa_anos'])
    porte = row['porte']
    sel = risco_df[
        (risco_df['cnae_section'] == sec) &
        (risco_df['porte']        == porte) &
        (risco_df['idade_min']    <= emp) &
        (risco_df['idade_max']    >= emp)
    ]
    return float(sel['risco_anual'].iloc[0]) if not sel.empty else 0.1002

def simulate_pricing(params):
    """
    Dada uma linha de parâmetros (PV, n, p_close_annual, p_rot_m, delay),
    retorna um dicionário com:
      • PMT_base, PMT_risco, spreads, TIR cliente
      • KPIs: EPVs, Expected_payments, Mean_time_to_event, E_Duration, LGD_ponderado
    """
    start = time.time()

    # --- 1) Extrai parâmetros básicos ----------------------------
    PV          = float(params['PV'])
    n           = int(params['n'])
    p_close_ann = float(params['p_close_annual'])
    p_rot_m     = float(params['p_rot_m'])
    d           = int(params['delay'])
    if n > 120:
        n = 120  # limite prático de parcelas

    # --- 2) Calcula hazards mensais -------------------------------
    # hazard de fechamento
    h_close    = 1 - (1 - p_close_ann)**(1/12)
    # turnover inclui fechamento + rotatividade
    h_turnover = h_close + (1 - h_close) * (p_rot_m * 0.66)
    # default = parte do turnover que vira inadimplência
    h_default  = h_turnover * P_INFORMALIDADE
    # delay = parte do turnover que só atrasa pagamentos
    h_delay    = h_turnover * (1 - P_INFORMALIDADE)

    # --- 3) Constrói sobrevivência e densidades ------------------
    S     = np.empty(n+1);    S[0] = 1.0
    f_def = np.zeros(n+1);    f_del = np.zeros(n+1)
    for u in range(1, n+1):
        S[u]     = S[u-1] * (1 - h_default - h_delay)
        f_def[u] = S[u-1] * h_default
        f_del[u] = S[u-1] * h_delay
    S_n = S[n]  # probabilidade de chegar ao fim sem evento

    # --- 4) PMT livre de risco (anuidade clássica) -------------
    PMT_base = PV * r_tar_m / (1 - (1 + r_tar_m)**(-n))

    # --- 5) EPV components e root‐finding para PMT_risco -------
    def epv_components(PMT):
        # saldo devedor ao longo do tempo
        bal = np.empty(n+1); bal[0] = PV
        for t in range(1, n+1):
            juros  = r_tar_m * bal[t-1]
            amort  = PMT - juros
            bal[t] = bal[t-1] - amort

        # EPV sobrevivência
        surv_pv = S[n] * sum(PMT / (1 + r_tar_m)**t for t in range(1, n+1))
        # EPV atraso (pagamentos remarcados em t+d)
        delay_pv = sum(
            f_del[u] * (
                sum(PMT / (1 + r_tar_m)**t       for t in range(1, u)) +
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
    spread   = r_min_m - r_tar_m
    spread_a = R_min_a - R_BASE

    # --- 7) KPIs adicionais -------------------------------------
    P_default_total    = f_def.sum()
    P_delay_total      = f_del.sum()
    Expected_payments  = S[:-1].sum()  # parcelas pagas **no prazo**
    Mean_time_to_event = sum(u * (f_def[u] + f_del[u]) for u in range(1, n+1))  #não pode ser calculado geometricamente!!!

    # **E(Duration)**: duração esperada do contrato
    E_Duration = (
        sum(u           * f_def[u] for u in range(1, n+1)) +
        sum((n + d)     * f_del[u] for u in range(1, n+1)) +
        n * S_n
    )

    # LGD ponderado
    bal = np.empty(n+1); bal[0] = PV
    for t in range(1, n+1):
        juros  = r_tar_m * bal[t-1]
        amort  = PMT_base - juros
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

def process_row(row, rot_df, cnae_col, riscos_df):
    sec = row.get('cnae_section') or 'A'
    if sec in rot_df[cnae_col].values:
        rot = rot_df.loc[rot_df[cnae_col] == sec].iloc[0]
    else:
        rot = rot_df.loc[rot_df[cnae_col] == 'A'].iloc[0]

    params = {
        'PV':             row['grossvalue'],
        'n':              row['numberofinstallments'],
        'p_close_annual': get_risco_fechamento_annual(row, riscos_df),
        'p_rot_m':        float(rot['rotatividade_mensal']),
        'delay':          int(rot['tempo_desemprego_esperado_meses'])
    }
    sim = simulate_pricing(params)
    return {
        'personid':     row['personid'],
        'contractid':   row['contractid'],
        'cnae_section': sec,
        'porte':        row.get('porte'),
        'delay':        params['delay'],
        'p_rot_m':      params['p_rot_m'],
        **sim
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num',    type=int)
    parser.add_argument('--sample',       type=int)
    parser.add_argument('-o', '--output', default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    rot_df    = pd.read_csv(CSV_ROTATIVIDADE, encoding='utf-8-sig')
    riscos_df = pd.read_csv(CSV_FECHAMENTO, encoding='latin1')
    base      = pd.read_csv(CSV_BASE,     encoding='latin1', dtype=str)
    for col in ['grossvalue','numberofinstallments','tempo_empresa_anos']:
        base[col] = base[col].astype(float)

    if args.num:
        base = base.head(args.num)
    if args.sample:
        base = base.sample(n=args.sample, random_state=42)

    records = base.to_dict('records')
    results = []
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
                  TimeElapsedColumn(), TimeRemainingColumn()) as prog:
        task = prog.add_task("Calculando pricing...", total=len(records))
        cnae_col = [c for c in rot_df.columns if 'cnae_section' in c][0]
        rot_df[cnae_col] = rot_df[cnae_col].astype(str)
        rot_df['rotatividade_mensal']              = rot_df['rotatividade_mensal'].astype(float)
        rot_df['tempo_desemprego_esperado_meses'] = rot_df['tempo_desemprego_esperado_meses'].astype(int)
        for row in records:
            results.append(process_row(row, rot_df, cnae_col, riscos_df))
            prog.update(task, advance=1)

    df = pd.DataFrame(results)

    # Ordena colunas conforme solicitado
    cols_order = [
        'personid','contractid','cnae_section','porte',
        'PV','n',
        'CDI_anual','Funding','Custo_Operacional','Margem','R_base_anual',
        'delay','p_rot_m',
        'h_close','h_turnover','h_default','h_delay',
        'S_n','P_default_total','P_delay_total',
        'EPV_surv','EPV_delay','EPV_default',
        'Expected_payments','Mean_time_to_event','E_Duration','LGD_ponderado',
        'PMT_base','PMT_risco','spread_valor',
        'r_tar_m','r_min_m','spread',
        'R_min_anual','spread_anual','calc_time_s'
    ]
    df = df[cols_order]

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"✅ Resultados salvos em {args.output}")
