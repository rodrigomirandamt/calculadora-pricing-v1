#!/usr/bin/env python
# validacao.py
# Script para validação dos resultados da calculadora de pricing

import pandas as pd
import numpy as np
import argparse
import sys

def validate_results(filepath):
    """
    Valida os resultados da calculadora de pricing
    
    Args:
        filepath: Caminho para o arquivo CSV de resultados
        
    Returns:
        str: Texto com resultados da validação
    """
    # 1) Carrega o arquivo CSV
    df = pd.read_csv(filepath, sep=',')

    # 2) Parâmetros de tolerância
    tol_prob  = 1e-6    # para soma de probabilidades
    tol_epv   = 0.05    # 5% de tolerância entre razão EPV/PV e S_n etc.
    tol_total = 1e-6    # para soma EPV = PV

    # 3) Calcula métricas auxiliares
    df['sum_probs']        = df['S_n'] + df['P_default_total'] + df['P_delay_total']
    df['epv_surv_ratio']   = df['EPV_surv']  / df['PV']
    df['epv_delay_ratio']  = df['EPV_delay'] / df['PV']
    df['epv_def_ratio']    = df['EPV_default']/ df['PV']
    df['epv_total']        = df['EPV_surv'] + df['EPV_delay'] + df['EPV_default']
    df['duration_check']   = df['E_Duration'] >= df['Mean_time_to_event']

    # 4) Flags de erro
    df['flag_sum_probs']   = (df['sum_probs'] - 1).abs() > tol_prob
    df['flag_epv_surv']    = (df['epv_surv_ratio'] - df['S_n']).abs() > tol_epv
    df['flag_epv_delay']   = (df['epv_delay_ratio'] - df['P_delay_total']).abs() > tol_epv
    df['flag_epv_def']     = (df['epv_def_ratio']   - df['P_default_total']).abs() > tol_epv
    df['flag_epv_total']   = (df['epv_total'] - df['PV']).abs() > tol_total
    df['flag_pmt']         = df['PMT_risco'] <= df['PMT_base']
    df['flag_duration']    = ~df['duration_check']
    df['flag_intervals']   = ~(
        (df['S_n'].between(0,1)) &
        (df['P_default_total'].between(0,1)) &
        (df['P_delay_total'].between(0,1)) &
        (df['LGD_ponderado'].between(0,1))
    )

    # 5) Resumo de flags
    checks = [
        'flag_sum_probs','flag_epv_surv','flag_epv_delay','flag_epv_def',
        'flag_epv_total','flag_pmt','flag_duration','flag_intervals'
    ]

    result_lines = []
    for chk in checks:
        n_bad = df[chk].sum()
        result_lines.append(f"{chk:20s}: {n_bad} linha(s) com problema")

    # 6) Exemplo: exibir as primeiras 10 linhas com qualquer flag True
    bad = df[df[checks].any(axis=1)]
    if not bad.empty:
        result_lines.append("\nLinhas com inconsistências:")
        result_lines.append(bad.head(10).to_string())
    else:
        result_lines.append("\nNenhuma inconsistência encontrada nos dados!")

    return '\n'.join(result_lines)

def main():
    parser = argparse.ArgumentParser(description='Validar resultados da calculadora de pricing')
    parser.add_argument('filepath', nargs='?', default='resultado_pricingv1.csv',
                        help='Caminho para o arquivo de resultados (CSV)')
    args = parser.parse_args()
    
    validation_results = validate_results(args.filepath)
    print(validation_results)

if __name__ == '__main__':
    main()
