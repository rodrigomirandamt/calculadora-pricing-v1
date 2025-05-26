#!/usr/bin/env python
# data_utils.py
# Funções para carregamento e processamento de dados

import pandas as pd
import numpy as np
from src.config import *

def load_data(base_path=CSV_BASE, rot_path=CSV_ROTATIVIDADE, risk_path=CSV_FECHAMENTO, 
              obito_path=CSV_OBITO, num_rows=None, sample_size=None):
    """
    Carrega e prepara os arquivos de dados de entrada
    
    Args:
        base_path: Caminho para o arquivo da base de contratos
        rot_path: Caminho para o arquivo de rotatividade
        risk_path: Caminho para o arquivo de riscos de fechamento
        obito_path: Caminho para o arquivo de probabilidades de óbito
        num_rows: Limite de linhas a carregar (opcional)
        sample_size: Tamanho da amostra aleatória (opcional)
        
    Returns:
        Tuple de DataFrames (base, rot_df, riscos_df, cnae_col)
    """
    rot_df = pd.read_csv(rot_path, encoding='utf-8-sig')
    riscos_df = pd.read_csv(risk_path, encoding='latin1')
    obito_df = pd.read_csv(obito_path, encoding='utf-8-sig')
    base = pd.read_csv(base_path, encoding='latin1', dtype=str)
    
    # Processa os dataframes
    for col in ['grossvalue', 'numberofinstallments', 'tempo_empresa_anos', 'valor_max_faturamento']:
        if col in base.columns:
            base[col] = base[col].astype(float)
    
    # Converte idade_anos para int truncando (não arredondando) para fazer o merge corretamente
    if 'idade_anos' in base.columns:
        base['idade_anos'] = np.floor(base['idade_anos'].astype(float)).astype(int)
    
    # Garante que idade_anos na tabela de óbito também seja int
    obito_df['idade_anos'] = obito_df['idade_anos'].astype(int)
    
    # Merge com dados de mortalidade
    base = base.merge(obito_df, on=['gender', 'idade_anos'], how='left')
    # Preenche valores faltantes com 0 (assumindo baixo risco para idades não cobertas)
    base['prob_anual_obito'] = base['prob_anual_obito'].fillna(0.0)
    
    cnae_col = [c for c in rot_df.columns if 'cnae_section' in c][0]
    rot_df[cnae_col] = rot_df[cnae_col].astype(str)
    rot_df['rotatividade_mensal'] = rot_df['rotatividade_mensal'].astype(float)
    rot_df['tempo_desemprego_esperado_meses'] = rot_df['tempo_desemprego_esperado_meses'].astype(int)
    
    # Aplica filtros, se especificados
    if num_rows:
        base = base.head(num_rows)
    if sample_size:
        base = base.sample(n=sample_size, random_state=42)
    
    return base, rot_df, riscos_df, cnae_col

def get_risco_fechamento_annual(row, risco_df):
    """
    Busca a probabilidade anual de fechamento pelo CNAE, tempo da empresa e faturamento.
    
    Args:
        row: Linha do DataFrame de contratos
        risco_df: DataFrame com os riscos de fechamento
        
    Returns:
        float: Probabilidade anual de fechamento
    """
    sec = row.get('cnae_section') or 'A'
    emp = float(row['tempo_empresa_anos'])
    faturamento = float(row.get('valor_max_faturamento', 0))
    
    sel = risco_df[
        (risco_df['cnae_section'] == sec) &
        (risco_df['idade_min'] <= emp) &
        (risco_df['idade_max'] >= emp) &
        (risco_df['faturamento_min'] <= faturamento) &
        (risco_df['faturamento_max'] >= faturamento)
    ]
    return float(sel['prob_ajustada'].iloc[0]) if not sel.empty else 0.1002

def process_row(row, rot_df, cnae_col, riscos_df):
    """
    Processa uma linha de contrato e calcula o pricing
    
    Args:
        row: Linha do DataFrame de contratos
        rot_df: DataFrame de rotatividade
        cnae_col: Nome da coluna de CNAE
        riscos_df: DataFrame de riscos
        
    Returns:
        dict: Dicionário com os resultados do cálculo de pricing
    """
    from src.pricing_model import simulate_pricing
    
    sec = row.get('cnae_section') or 'A'
    if sec in rot_df[cnae_col].values:
        rot = rot_df.loc[rot_df[cnae_col] == sec].iloc[0]
    else:
        rot = rot_df.loc[rot_df[cnae_col] == 'A'].iloc[0]

    p_close_annual = get_risco_fechamento_annual(row, riscos_df)

    params = {
        'PV': row['grossvalue'],
        'n': row['numberofinstallments'],
        'p_close_annual': p_close_annual,
        'p_rot_m': float(rot['rotatividade_mensal']),
        'delay': int(rot['tempo_desemprego_esperado_meses']),
        'p_obito_ann': float(row.get('prob_anual_obito', 0.0))
    }
    sim = simulate_pricing(params)
    return {
        'personid': row['personid'],
        'contractid': row['contractid'],
        'cnae_section': sec,
        'porte': row.get('porte'),
        'valor_max_faturamento': row.get('valor_max_faturamento', 0),
        'tempo_empresa_anos': row.get('tempo_empresa_anos', 0),
        'cluster_person': row.get('cluster_person', ''),
        'idade': row.get('idade_anos', 0),
        'sexo': row.get('gender', ''),
        'p_close_annual': p_close_annual,
        'delay': params['delay'],
        'p_rot_m': params['p_rot_m'],
        'p_obito_ann': params['p_obito_ann'],
        **sim
    } 