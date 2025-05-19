#!/usr/bin/env python
# simulador.py
# Simulador interativo de pricing de empréstimos consignados

import streamlit as st
import pandas as pd
import numpy as np
from src.data_utils import load_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib
import src.config as config_module
import src.pricing_model as pricing_model_module

# Configuração da página
st.set_page_config(
    page_title="Simulador de Pricing",
    page_icon="💰",
    layout="wide"
)

# Título e descrição
st.title("💰 Simulador de Pricing - Consignado Privado")
st.markdown("""
Este simulador calcula o pricing ajustado ao risco de empréstimos consignados privados,
considerando características da empresa e parâmetros financeiros.
""")

# Carrega dados de referência
@st.cache_data
def load_reference_data():
    _, rot_df, riscos_df, cnae_col = load_data()
    return rot_df, riscos_df, cnae_col

rot_df, riscos_df, cnae_col = load_reference_data()

# Layout em duas colunas
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Características da Empresa")
    
    # Seleção de CNAE
    cnae_options = sorted(rot_df[cnae_col].unique())
    cnae_section = st.selectbox(
        "Setor (CNAE)",
        options=cnae_options,
        index=0
    )
    
    # Características da empresa
    tempo_empresa_anos = st.slider(
        "Idade da Empresa (anos)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )
    
    valor_max_faturamento = st.number_input(
        "Faturamento Anual (R$)",
        min_value=0.0,
        max_value=10000000.0,
        value=1200000.0,
        step=100000.0,
        format="%.2f"
    )

with col2:
    st.subheader("💰 Parâmetros Financeiros")
    
    # Parâmetros de risco
    P_INFORMALIDADE_ui = st.slider(
        "Taxa de Informalidade (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0
    ) / 100
    
    LGD_ui = st.slider(
        "Loss Given Default - LGD (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    ) / 100
    
    # Parâmetros financeiros
    CDI_ANUAL_ui = st.slider(
        "CDI Anual (%)",
        min_value=0.0,
        max_value=30.0,
        value=14.0,
        step=0.1
    ) / 100
    
    # O slider "FUNDING" é, na verdade, o SPREAD sobre o CDI
    FUNDING_spread_ui = st.slider(
        "Spread de Funding (%)", # Este é o spread
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    ) / 100
    
    CUSTO_OPERACIONAL_ui = st.slider(
        "Custo Operacional (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1
    ) / 100
    
    MARGEM_ui = st.slider(
        "Margem (%)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.1
    ) / 100

# Características do empréstimo
st.subheader("📝 Características do Empréstimo")
col3, col4 = st.columns(2)

with col3:
    PV_ui = st.number_input(
        "Valor do Empréstimo (R$)",
        min_value=1000.0,
        max_value=1000000.0,
        value=50000.0,
        step=1000.0,
        format="%.2f"
    )

with col4:
    n_ui = st.slider(
        "Prazo (meses)",
        min_value=12,
        max_value=120,
        value=36,
        step=12
    )

# Busca dados de rotatividade para o CNAE selecionado
rot_data = rot_df[rot_df[cnae_col] == cnae_section].iloc[0]
p_rot_m = float(rot_data['rotatividade_mensal'])
delay = int(rot_data['tempo_desemprego_esperado_meses'])

# Busca risco de fechamento
sel = riscos_df[
    (riscos_df['cnae_section'] == cnae_section) &
    (riscos_df['idade_min'] <= tempo_empresa_anos) &
    (riscos_df['idade_max'] >= tempo_empresa_anos) &
    (riscos_df['faturamento_min'] <= valor_max_faturamento) &
    (riscos_df['faturamento_max'] >= valor_max_faturamento)
]
p_close_annual = float(sel['prob_ajustada'].iloc[0]) if not sel.empty else 0.1002

# --- INÍCIO DA LÓGICA CRÍTICA DE ATUALIZAÇÃO E CÁLCULO ---

# 1. Recarrega o módulo src.config (redefine suas variáveis para os padrões do arquivo)
importlib.reload(config_module)

# 2. Define os atributos do módulo config_module (src.config) com os valores dos sliders
config_module.CDI_ANUAL = CDI_ANUAL_ui
# Calcula a taxa de FUNDING real (CDI + Spread) e a define no módulo
actual_funding_rate = CDI_ANUAL_ui + FUNDING_spread_ui
config_module.FUNDING = actual_funding_rate
config_module.CUSTO_OPERACIONAL = CUSTO_OPERACIONAL_ui
config_module.MARGEM = MARGEM_ui

# 3. RECALCULA R_BASE e r_tar_m DENTRO do namespace do módulo config_module
#    usando os valores atualizados de FUNDING, CUSTO_OPERACIONAL, MARGEM
config_module.R_BASE = config_module.FUNDING + config_module.CUSTO_OPERACIONAL + config_module.MARGEM
config_module.r_tar_m = (1 + config_module.R_BASE)**(1/12) - 1

config_module.P_INFORMALIDADE = P_INFORMALIDADE_ui
config_module.LGD = LGD_ui

# 4. Recarrega o módulo src.pricing_model para que ele use o src.config atualizado
importlib.reload(pricing_model_module)

# 5. Importa a função simulate_pricing DEPOIS que pricing_model_module foi recarregado
#    Isso garante que estamos usando a versão mais recente da função que "vê" o config atualizado.
from src.pricing_model import simulate_pricing

# Prepara parâmetros para a simulação
params_for_simulation = {
    'PV': PV_ui,
    'n': n_ui,
    'p_close_annual': p_close_annual, # Este vem dos dados, não do slider
    'p_rot_m': p_rot_m,               # Este vem dos dados, não do slider
    'delay': delay                    # Este vem dos dados, não do slider
}

# Executa a simulação
result = simulate_pricing(params_for_simulation)

# --- FIM DA LÓGICA CRÍTICA ---

# Exibe resultados
st.subheader("📈 Resultados do Pricing")

# Layout em 3 colunas para os resultados
col5, col6, col7 = st.columns(3)

# Calcula p_rot_anual
p_rot_anual = 1 - (1 - p_rot_m)**12

with col5:
    st.metric("Probabilidade Anual de Fechamento", f"{p_close_annual:.2%}")
    st.metric("Prob. Rotatividade Anual", f"{p_rot_anual:.2%}")
    st.metric("Taxa Base Anual", f"{result['R_base_anual']:.2%}")

with col6:
    st.metric("PMT Base", f"R$ {result['PMT_base']:,.2f}")
    st.metric("PMT Risco", f"R$ {result['PMT_risco']:,.2f}")
    st.metric("Spread Valor", f"R$ {result['spread_valor']:,.2f}")

with col7:
    st.metric("Taxa Mínima Mensal", f"{result['r_min_m']:.2%}")
    st.metric("Taxa Mínima Anual", f"{result['R_min_anual']:.2%}")
    st.metric("Spread Anual", f"{result['spread_anual']:.2%}")

# Gráficos lado a lado usando colunas do Streamlit
col_pie, col_bar = st.columns(2)

with col_pie:
    st.plotly_chart(
        go.Figure(
            go.Pie(
                labels=['Sobrevivência', 'Atraso', 'Default'],
                values=[result['EPV_surv'], result['EPV_delay'], result['EPV_default']],
                hole=.3
            )
        ).update_layout(title="Composição do EPV"),
        use_container_width=True
    )

with col_bar:
    st.plotly_chart(
        go.Figure(
            go.Bar(
                x=['Default Total', 'Atraso Total', 'Sobrevivência'],
                y=[result['P_default_total'], result['P_delay_total'], result['S_n']],
                text=[f"{v:.1%}" for v in [result['P_default_total'], result['P_delay_total'], result['S_n']]],
                textposition='auto',
            )
        ).update_layout(title="Probabilidades de Evento"),
        use_container_width=True
    )

# Métricas adicionais
st.subheader("📊 Métricas Adicionais")
col8, col9, col10 = st.columns(3)

with col8:
    st.metric("Parcelas Esperadas", f"{result['Expected_payments']:.1f}")
    st.metric("Tempo Médio até Evento", f"{result['Mean_time_to_event']:.1f} meses")

with col9:
    st.metric("Duração Esperada", f"{result['E_Duration']:.1f} meses")
    st.metric("LGD Ponderado", f"{result['LGD_ponderado']:.2%}")

with col10:
    st.metric("Tempo de Cálculo", f"{result['calc_time_s']:.3f} segundos")
