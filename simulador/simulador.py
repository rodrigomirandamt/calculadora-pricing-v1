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

# --- INSTRUÇÕES PARA SALVAR COMO PDF ---
st.info("""
**🖨️ Para salvar esta página como PDF:**
1. Use o atalho **Ctrl+P** (Windows/Linux) ou **Cmd+P** (Mac)
2. Selecione **Salvar como PDF** (ou equivalente) como impressora/destino
3. Clique em Salvar/Imprimir
""")

# Carrega dados de referência
@st.cache_data
def load_reference_data():
    _, rot_df, riscos_df, cnae_col = load_data()
    return rot_df, riscos_df, cnae_col

rot_df, riscos_df, cnae_col = load_reference_data()

# Layout otimizado em três colunas
col1, col2, col3 = st.columns([1, 1, 1])

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
    
    st.subheader("👤 Características do Tomador")
    
    idade_ui = st.slider(
        "Idade (anos)",
        min_value=21,
        max_value=75,
        value=35,
        step=1
    )
    
    sexo_ui = st.selectbox(
        "Sexo",
        options=['M', 'F'],
        index=0,
        format_func=lambda x: 'Masculino' if x == 'M' else 'Feminino'
    )

with col2:
    st.subheader("💰 Parâmetros Financeiros")
    
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

with col3:
    st.subheader("⚠️ Parâmetros de Risco")
    
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
    
    st.subheader("📝 Características do Empréstimo")
    
    PV_ui = st.number_input(
        "Valor do Empréstimo (R$)",
        min_value=1000.0,
        max_value=1000000.0,
        value=50000.0,
        step=1000.0,
        format="%.2f"
    )
    
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

# Carrega dados de mortalidade
@st.cache_data
def load_mortality_data():
    import pandas as pd
    return pd.read_csv('dados/prob-anual-obito.csv', encoding='utf-8-sig')

obito_df = load_mortality_data()

# Busca probabilidade de óbito
obito_sel = obito_df[
    (obito_df['gender'] == sexo_ui) &
    (obito_df['idade_anos'] == idade_ui)
]
p_obito_ann = float(obito_sel['prob_anual_obito'].iloc[0]) if not obito_sel.empty else 0.0

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
    'delay': delay,                   # Este vem dos dados, não do slider
    'p_obito_ann': p_obito_ann        # Este vem dos dados de mortalidade
}

# Executa a simulação
result = simulate_pricing(params_for_simulation)

# --- FIM DA LÓGICA CRÍTICA ---

# Exibe resultados
st.subheader("📈 Resultados do Pricing")

# Calcula p_rot_anual
p_rot_anual = 1 - (1 - p_rot_m)**12

# Layout em 4 colunas para os resultados principais
col_res1, col_res2, col_res3, col_res4 = st.columns(4)

with col_res1:
    st.metric("Prob. Fechamento Anual", f"{p_close_annual:.2%}", help="Probabilidade estimada de a empresa encerrar suas atividades em um ano, com base no CNAE, idade e faturamento.")
    st.metric("Prob. Rotatividade Anual", f"{p_rot_anual:.2%}", help="Probabilidade estimada de o tomador do empréstimo ser desligado da empresa em um ano (inclui risco de fechamento da empresa e turnover individual).")
    st.metric("Prob. Óbito Anual", f"{p_obito_ann:.2%}", help="Probabilidade anual de óbito do tomador com base na idade e gênero.")

with col_res2:
    st.metric("Taxa Base Anual", f"{result['R_base_anual']:.2%}", help="Taxa de juros anual que remunera o capital, cobre custos e margem, sem considerar o risco de crédito. R_BASE = Funding + Custo Operacional + Margem")
    st.metric("PMT Base", f"R$ {result['PMT_base']:,.2f}", help="Parcela mensal calculada apenas com a Taxa Base, sem ajuste para riscos de default ou atraso.")
    st.metric("PMT Risco", f"R$ {result['PMT_risco']:,.2f}", help="Parcela mensal ajustada para cobrir os riscos esperados de default e atraso, garantindo que o Valor Presente Esperado (EPV) dos fluxos de caixa iguale o Valor do Empréstimo (PV).")

with col_res3:
    st.metric("Spread Valor", f"R$ {result['spread_valor']:,.2f}", help="Diferença monetária entre a PMT Risco e a PMT Base (PMT_Risco - PMT_Base).")
    st.metric("Taxa Mínima Mensal", f"{result['r_min_m']:.2%}", help="Taxa Interna de Retorno (TIR) mensal efetiva para o cliente, considerando a PMT Risco.")
    st.metric("Taxa Mínima Anual", f"{result['R_min_anual']:.2%}", help="Taxa Interna de Retorno (TIR) anual efetiva para o cliente, (1 + r_min_m)^12 - 1.")

with col_res4:
    st.metric("Spread Anual", f"{result['spread_anual']:.2%}", help="Diferença entre a Taxa Mínima Anual (R_min_anual) e a Taxa Base Anual (R_base_anual). Representa o prêmio de risco anual.")
    st.metric("Parcelas Esperadas", f"{result['Expected_payments']:.1f}", help="Número esperado de parcelas que serão pagas integralmente e no prazo, considerando as probabilidades de sobrevivência em cada período.")
    st.metric("Tempo Médio até Evento", f"{result['Mean_time_to_event']:.1f} meses", help="Número médio de meses até a ocorrência do primeiro evento de crédito (default ou atraso).")

# Métricas adicionais em layout compacto
st.subheader("📊 Métricas Adicionais")
col_add1, col_add2, col_add3, col_add4 = st.columns(4)

with col_add1:
    st.metric("Duração Esperada", f"{result['E_Duration']:.1f} meses", help="Prazo médio ponderado pelos eventos. Considera o tempo até default, tempo até o fim do contrato após um atraso (incluindo o período de delay 'd'), e o prazo total para contratos sem evento.")

with col_add2:
    st.metric("LGD Ponderado", f"{result['LGD_ponderado']:.2%}", help="Perda Esperada Total (EL) como percentual do PV. Calculado como a soma das (Probabilidade de Default em cada mês * LGD * Saldo Devedor do mês) / PV.")

with col_add3:
    st.metric("LGD Ponderado Óbito", f"{result['LGD_ponderado_death']:.2%}", help="Perda Esperada por Óbito como percentual do PV. Calculado como a soma das (Probabilidade de Óbito em cada mês * LGD_DEATH * Saldo Devedor do mês) / PV.")

with col_add4:
    st.metric("Tempo de Cálculo", f"{result['calc_time_s']:.3f} segundos", help="Tempo gasto para executar a simulação de pricing.")

# Gráficos principais em layout otimizado
st.subheader("📊 Visualizações dos Resultados")

# Gráficos lado a lado usando colunas do Streamlit
col_pie, col_bar = st.columns(2)

with col_pie:
    st.plotly_chart(
        go.Figure(
            go.Pie(
                labels=['Sobrevivência', 'Atraso', 'Default', 'Óbito'],
                values=[result['EPV_surv'], result['EPV_delay'], result['EPV_default'], result['EPV_death']],
                hole=.3
            )
        ).update_layout(
            title_text="Composição do EPV (Valor Presente Esperado dos Fluxos)", 
            title_x=0.5,
            margin=dict(t=60, b=20, l=20, r=20),
            height=400
        ),
        use_container_width=True
    )

with col_bar:
    st.plotly_chart(
        go.Figure(
            go.Bar(
                x=['Default Total', 'Atraso Total', 'Óbito Total', 'Sobrevivência até o Fim'],
                y=[result['P_default_total'], result['P_delay_total'], result['P_death_total'], result['S_n']],
                text=[f"{v:.1%}" for v in [result['P_default_total'], result['P_delay_total'], result['P_death_total'], result['S_n']]],
                textposition='auto',
                marker_color=['red', 'orange', 'purple', 'green']
            )
        ).update_layout(
            title_text="Probabilidades Acumuladas de Evento ao Final do Prazo", 
            title_x=0.5,
            margin=dict(t=60, b=20, l=20, r=20),
            height=400
        ),
        use_container_width=True
    )

# --- GRÁFICO DE SENSIBILIDADE: SPREAD ANUAL POR CNAE ---
st.subheader("📉 Sensibilidade do Spread Anual ao CNAE")

# Lista de CNAEs disponíveis
cnae_sens_options = cnae_options
spread_anual_list = []

# Para cada CNAE, simular spread anual mantendo os demais parâmetros fixos
for cnae in cnae_sens_options:
    # Busca dados de rotatividade para o CNAE
    rot_data_sens = rot_df[rot_df[cnae_col] == cnae].iloc[0]
    p_rot_m_sens = float(rot_data_sens['rotatividade_mensal'])
    delay_sens = int(rot_data_sens['tempo_desemprego_esperado_meses'])
    # Busca risco de fechamento para o CNAE, idade e faturamento selecionados
    sel_sens = riscos_df[
        (riscos_df['cnae_section'] == cnae) &
        (riscos_df['idade_min'] <= tempo_empresa_anos) &
        (riscos_df['idade_max'] >= tempo_empresa_anos) &
        (riscos_df['faturamento_min'] <= valor_max_faturamento) &
        (riscos_df['faturamento_max'] >= valor_max_faturamento)
    ]
    p_close_annual_sens = float(sel_sens['prob_ajustada'].iloc[0]) if not sel_sens.empty else 0.1002
    # Parâmetros para simulação
    params_sens = {
        'PV': PV_ui,
        'n': n_ui,
        'p_close_annual': p_close_annual_sens,
        'p_rot_m': p_rot_m_sens,
        'delay': delay_sens
    }
    # Rodar simulação
    try:
        result_sens = simulate_pricing(params_sens)
        spread_anual_list.append(result_sens['spread_anual'])
    except Exception as e:
        spread_anual_list.append(np.nan)

# Montar DataFrame para plotagem
sens_df = pd.DataFrame({
    'CNAE': cnae_sens_options,
    'Spread Anual (%)': [100 * s if pd.notnull(s) else np.nan for s in spread_anual_list]
})
sens_df = sens_df.sort_values('Spread Anual (%)', ascending=False)

# Identificar índice do CNAE selecionado
idx_cnae = sens_df['CNAE'].tolist().index(cnae_section) if cnae_section in sens_df['CNAE'].tolist() else None
spread_atual_cnae = sens_df['Spread Anual (%)'].iloc[idx_cnae] if idx_cnae is not None else None

# Gráfico de barras horizontais
fig_sens = go.Figure()
fig_sens.add_trace(
    go.Bar(
        x=sens_df['Spread Anual (%)'],
        y=sens_df['CNAE'],
        orientation='h',
        marker_color='royalblue',
        text=[f"{v:.2f}%" if pd.notnull(v) else "" for v in sens_df['Spread Anual (%)']],
        textposition='auto',
        name='Spread Anual'
    )
)
# Adiciona marcador especial para o CNAE selecionado
if idx_cnae is not None and spread_atual_cnae is not None and not pd.isna(spread_atual_cnae):
    fig_sens.add_trace(
        go.Scatter(
            x=[spread_atual_cnae],
            y=[cnae_section],
            mode='markers+text',
            marker=dict(size=18, color='firebrick', symbol='diamond'),
            text=[f"Selecionado: {cnae_section}"],
            textposition='middle right',
            name='Selecionado',
            showlegend=False
        )
    )
fig_sens.update_layout(
    title="Sensibilidade do Spread Anual ao CNAE",
    xaxis_title="Spread Anual (%)",
    yaxis_title="CNAE",
    height=400 + 10 * len(sens_df),
    margin=dict(l=120, r=40, t=60, b=40)
)
st.plotly_chart(fig_sens, use_container_width=True)

# --- GRÁFICOS DE SENSIBILIDADE: IDADE DA EMPRESA E FATURAMENTO ---
st.subheader("📈 Sensibilidade do Spread Anual - Características da Empresa")

# Layout em duas colunas para os gráficos da empresa
col_idade_emp, col_faturamento = st.columns(2)

with col_idade_emp:
    # Faixa de idades para simulação
    idade_range = np.arange(0, 51, 1)
    spread_anual_idade = []

    for idade in idade_range:
        # Usar o CNAE atualmente selecionado
        rot_data_idade = rot_df[rot_df[cnae_col] == cnae_section].iloc[0]
        p_rot_m_idade = float(rot_data_idade['rotatividade_mensal'])
        delay_idade = int(rot_data_idade['tempo_desemprego_esperado_meses'])
        # Busca risco de fechamento para o CNAE, idade e faturamento selecionados
        sel_idade = riscos_df[
            (riscos_df['cnae_section'] == cnae_section) &
            (riscos_df['idade_min'] <= idade) &
            (riscos_df['idade_max'] >= idade) &
            (riscos_df['faturamento_min'] <= valor_max_faturamento) &
            (riscos_df['faturamento_max'] >= valor_max_faturamento)
        ]
        p_close_annual_idade = float(sel_idade['prob_ajustada'].iloc[0]) if not sel_idade.empty else 0.1002
        params_idade = {
            'PV': PV_ui,
            'n': n_ui,
            'p_close_annual': p_close_annual_idade,
            'p_rot_m': p_rot_m_idade,
            'delay': delay_idade
        }
        try:
            result_idade = simulate_pricing(params_idade)
            spread_anual_idade.append(result_idade['spread_anual'] * 100)
        except Exception as e:
            spread_anual_idade.append(np.nan)

    # Valor atualmente selecionado na interface
    idade_atual = int(round(tempo_empresa_anos))
    if 0 <= idade_atual <= 50:
        spread_atual_idade = spread_anual_idade[idade_atual]
    else:
        spread_atual_idade = None

    fig_idade = go.Figure()
    fig_idade.add_trace(
        go.Scatter(
            x=idade_range,
            y=spread_anual_idade,
            mode='lines+markers',
            line=dict(color='firebrick', width=3),
            marker=dict(size=4),
            name='Spread Anual'
        )
    )
    # Adiciona marcador especial para o valor selecionado
    if spread_atual_idade is not None and not np.isnan(spread_atual_idade):
        fig_idade.add_trace(
            go.Scatter(
                x=[idade_atual],
                y=[spread_atual_idade],
                mode='markers+text',
                marker=dict(size=12, color='royalblue', symbol='diamond'),
                text=[f"{idade_atual} anos"],
                textposition='top center',
                name='Selecionado',
                showlegend=False
            )
        )
    fig_idade.update_layout(
        title="Idade da Empresa",
        xaxis_title="Idade da Empresa (anos)",
        yaxis_title="Spread Anual (%)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_idade, use_container_width=True)

with col_faturamento:
    # Faixa de faturamento para simulação (escala logarítmica, mais pontos)
    faturamento_range = np.logspace(np.log10(3e5), np.log10(1e9), 20)
    spread_anual_faturamento = []

    for faturamento in faturamento_range:
        # Usar o CNAE e idade atualmente selecionados
        rot_data_fat = rot_df[rot_df[cnae_col] == cnae_section].iloc[0]
        p_rot_m_fat = float(rot_data_fat['rotatividade_mensal'])
        delay_fat = int(rot_data_fat['tempo_desemprego_esperado_meses'])
        # Busca risco de fechamento para o CNAE, idade e faturamento
        sel_fat = riscos_df[
            (riscos_df['cnae_section'] == cnae_section) &
            (riscos_df['idade_min'] <= tempo_empresa_anos) &
            (riscos_df['idade_max'] >= tempo_empresa_anos) &
            (riscos_df['faturamento_min'] <= faturamento) &
            (riscos_df['faturamento_max'] >= faturamento)
        ]
        p_close_annual_fat = float(sel_fat['prob_ajustada'].iloc[0]) if not sel_fat.empty else 0.1002
        params_fat = {
            'PV': PV_ui,
            'n': n_ui,
            'p_close_annual': p_close_annual_fat,
            'p_rot_m': p_rot_m_fat,
            'delay': delay_fat
        }
        try:
            result_fat = simulate_pricing(params_fat)
            spread_anual_faturamento.append(result_fat['spread_anual'] * 100)
        except Exception as e:
            spread_anual_faturamento.append(np.nan)

    # Formatar os valores de faturamento para o eixo x (labels principais)
    faturamento_ticks = np.array([3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9])
    faturamento_labels = [
        'R$ 300k', 'R$ 1M', 'R$ 3M', 'R$ 10M', 'R$ 30M', 'R$ 100M', 'R$ 300M', 'R$ 1B'
    ]

    # Valor atualmente selecionado na interface
    faturamento_atual = valor_max_faturamento
    # Encontrar o ponto mais próximo na escala
    idx_fat = (np.abs(faturamento_range - faturamento_atual)).argmin()
    spread_atual_fat = spread_anual_faturamento[idx_fat] if 0 <= idx_fat < len(spread_anual_faturamento) else None

    fig_fat = go.Figure()
    fig_fat.add_trace(
        go.Scatter(
            x=faturamento_range,
            y=spread_anual_faturamento,
            mode='lines+markers',
            line=dict(color='seagreen', width=3),
            marker=dict(size=4),
            name='Spread Anual'
        )
    )
    # Adiciona marcador especial para o valor selecionado
    if spread_atual_fat is not None and not np.isnan(spread_atual_fat):
        fig_fat.add_trace(
            go.Scatter(
                x=[faturamento_range[idx_fat]],
                y=[spread_atual_fat],
                mode='markers+text',
                marker=dict(size=12, color='firebrick', symbol='diamond'),
                text=[f"R$ {faturamento_labels[np.abs(faturamento_ticks - faturamento_range[idx_fat]).argmin()]}"],
                textposition='top center',
                name='Selecionado',
                showlegend=False
            )
        )
    fig_fat.update_layout(
        title="Faturamento Anual",
        xaxis=dict(
            title="Faturamento Anual",
            tickvals=faturamento_ticks,
            ticktext=faturamento_labels,
            type='log'
        ),
        yaxis_title="Spread Anual (%)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_fat, use_container_width=True)

# --- GRÁFICOS DE SENSIBILIDADE: PARÂMETROS DE RISCO ---
st.subheader("📈 Sensibilidade do Spread Anual - Parâmetros de Risco")

# Layout em duas colunas para os parâmetros de risco
col_informalidade, col_lgd = st.columns(2)

with col_informalidade:
    informalidade_range = np.linspace(0, 1, 21)  # 0% a 100%, passo de 5%
    spread_anual_informalidade = []

    for p_inf_loop_val in informalidade_range:
        # Recarregar config_module para resetar para os valores do arquivo config.py
        importlib.reload(config_module)
        # Aplicar TODAS as configurações da UI, exceto a que está sendo variada
        config_module.CDI_ANUAL = CDI_ANUAL_ui
        actual_funding_rate_sens = CDI_ANUAL_ui + FUNDING_spread_ui
        config_module.FUNDING = actual_funding_rate_sens
        config_module.CUSTO_OPERACIONAL = CUSTO_OPERACIONAL_ui
        config_module.MARGEM = MARGEM_ui
        config_module.R_BASE = config_module.FUNDING + config_module.CUSTO_OPERACIONAL + config_module.MARGEM
        config_module.r_tar_m = (1 + config_module.R_BASE)**(1/12) - 1
        config_module.LGD = LGD_ui # Usar LGD da UI

        # Aplicar o valor da informalidade para esta iteração do loop
        config_module.P_INFORMALIDADE = p_inf_loop_val
        
        # Recarregar pricing_model para usar as configs atualizadas
        importlib.reload(pricing_model_module)
        from src.pricing_model import simulate_pricing

        rot_data_inf = rot_df[rot_df[cnae_col] == cnae_section].iloc[0]
        p_rot_m_inf = float(rot_data_inf['rotatividade_mensal'])
        delay_inf = int(rot_data_inf['tempo_desemprego_esperado_meses'])
        sel_inf = riscos_df[
            (riscos_df['cnae_section'] == cnae_section) &
            (riscos_df['idade_min'] <= tempo_empresa_anos) &
            (riscos_df['idade_max'] >= tempo_empresa_anos) &
            (riscos_df['faturamento_min'] <= valor_max_faturamento) &
            (riscos_df['faturamento_max'] >= valor_max_faturamento)
        ]
        p_close_annual_inf = float(sel_inf['prob_ajustada'].iloc[0]) if not sel_inf.empty else 0.1002
        params_inf = {
            'PV': PV_ui,
            'n': n_ui,
            'p_close_annual': p_close_annual_inf,
            'p_rot_m': p_rot_m_inf,
            'delay': delay_inf
        }
        try:
            result_inf = simulate_pricing(params_inf)
            spread_anual_informalidade.append(result_inf['spread_anual'] * 100)
        except Exception as e:
            spread_anual_informalidade.append(np.nan)

    # Valor atualmente selecionado na interface
    informalidade_atual = P_INFORMALIDADE_ui
    idx_inf = (np.abs(informalidade_range - informalidade_atual)).argmin()
    spread_atual_inf = spread_anual_informalidade[idx_inf] if 0 <= idx_inf < len(spread_anual_informalidade) else None

    fig_inf = go.Figure()
    fig_inf.add_trace(
        go.Scatter(
            x=informalidade_range * 100,
            y=spread_anual_informalidade,
            mode='lines+markers',
            line=dict(color='purple', width=3),
            marker=dict(size=4),
            name='Spread Anual'
        )
    )
    # Adiciona marcador especial para o valor selecionado
    if spread_atual_inf is not None and not np.isnan(spread_atual_inf):
        fig_inf.add_trace(
            go.Scatter(
                x=[informalidade_range[idx_inf] * 100],
                y=[spread_atual_inf],
                mode='markers+text',
                marker=dict(size=12, color='orange', symbol='diamond'),
                text=[f"{int(round(informalidade_range[idx_inf]*100))}%"],
                textposition='top center',
                name='Selecionado',
                showlegend=False
            )
        )
    fig_inf.update_layout(
        title="Taxa de Informalidade",
        xaxis_title="Taxa de Informalidade (%)",
        yaxis_title="Spread Anual (%)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_inf, use_container_width=True)

with col_lgd:
    lgd_range = np.linspace(0, 1, 21)  # 0% a 100%, passo de 5%
    spread_anual_lgd = []

    for lgd_loop_val in lgd_range:
        # Recarregar config_module para resetar para os valores do arquivo config.py
        importlib.reload(config_module)
        # Aplicar TODAS as configurações da UI, exceto a que está sendo variada
        config_module.CDI_ANUAL = CDI_ANUAL_ui
        actual_funding_rate_sens_lgd = CDI_ANUAL_ui + FUNDING_spread_ui
        config_module.FUNDING = actual_funding_rate_sens_lgd
        config_module.CUSTO_OPERACIONAL = CUSTO_OPERACIONAL_ui
        config_module.MARGEM = MARGEM_ui
        config_module.R_BASE = config_module.FUNDING + config_module.CUSTO_OPERACIONAL + config_module.MARGEM
        config_module.r_tar_m = (1 + config_module.R_BASE)**(1/12) - 1
        config_module.P_INFORMALIDADE = P_INFORMALIDADE_ui # Usar P_INFORMALIDADE da UI

        # Aplicar o valor de LGD para esta iteração do loop
        config_module.LGD = lgd_loop_val
        
        # Recarregar pricing_model para usar as configs atualizadas
        importlib.reload(pricing_model_module)
        from src.pricing_model import simulate_pricing

        rot_data_lgd = rot_df[rot_df[cnae_col] == cnae_section].iloc[0]
        p_rot_m_lgd = float(rot_data_lgd['rotatividade_mensal'])
        delay_lgd = int(rot_data_lgd['tempo_desemprego_esperado_meses'])
        sel_lgd = riscos_df[
            (riscos_df['cnae_section'] == cnae_section) &
            (riscos_df['idade_min'] <= tempo_empresa_anos) &
            (riscos_df['idade_max'] >= tempo_empresa_anos) &
            (riscos_df['faturamento_min'] <= valor_max_faturamento) &
            (riscos_df['faturamento_max'] >= valor_max_faturamento)
        ]
        p_close_annual_lgd = float(sel_lgd['prob_ajustada'].iloc[0]) if not sel_lgd.empty else 0.1002
        params_lgd = {
            'PV': PV_ui,
            'n': n_ui,
            'p_close_annual': p_close_annual_lgd,
            'p_rot_m': p_rot_m_lgd,
            'delay': delay_lgd
        }
        try:
            result_lgd = simulate_pricing(params_lgd)
            spread_anual_lgd.append(result_lgd['spread_anual'] * 100)
        except Exception as e:
            spread_anual_lgd.append(np.nan)

    # Valor atualmente selecionado na interface
    lgd_atual = LGD_ui
    idx_lgd = (np.abs(lgd_range - lgd_atual)).argmin()
    spread_atual_lgd = spread_anual_lgd[idx_lgd] if 0 <= idx_lgd < len(spread_anual_lgd) else None

    fig_lgd = go.Figure()
    fig_lgd.add_trace(
        go.Scatter(
            x=lgd_range * 100,
            y=spread_anual_lgd,
            mode='lines+markers',
            line=dict(color='darkgreen', width=3),
            marker=dict(size=4),
            name='Spread Anual'
        )
    )
    # Adiciona marcador especial para o valor selecionado
    if spread_atual_lgd is not None and not np.isnan(spread_atual_lgd):
        fig_lgd.add_trace(
            go.Scatter(
                x=[lgd_range[idx_lgd] * 100],
                y=[spread_atual_lgd],
                mode='markers+text',
                marker=dict(size=12, color='orange', symbol='diamond'),
                text=[f"{int(round(lgd_range[idx_lgd]*100))}%"],
                textposition='top center',
                name='Selecionado',
                showlegend=False
            )
        )
    fig_lgd.update_layout(
        title="LGD (Loss Given Default)",
        xaxis_title="LGD (%)",
        yaxis_title="Spread Anual (%)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_lgd, use_container_width=True)

# --- GRÁFICO DE SENSIBILIDADE: SPREAD ANUAL POR R_BASE ---
st.subheader("📊 Sensibilidade do Spread Anual à Taxa Base Anual (R_BASE = Funding + Custo Op. + Margem)")

# Faixa de R_BASE para simulação (anual)
r_base_range = np.arange(0.05, 0.501, 0.05)  # 5% a 50%, passo de 5%
spread_anual_rbase = []

for r_base_iter in r_base_range:
    # Recarregar config_module para resetar para os valores do arquivo config.py
    importlib.reload(config_module)
    
    # Aplicar configurações financeiras da UI, mas sobrescrever R_BASE e r_tar_m
    config_module.CDI_ANUAL = CDI_ANUAL_ui # Mantido para consistência, embora não afete R_BASE diretamente aqui
    # Os componentes individuais (FUNDING, CUSTO, MARGEM) não são usados para calcular R_BASE nesta simulação específica
    # Define o R_BASE da iteração
    config_module.R_BASE = r_base_iter
    # Recalcula r_tar_m com base no R_BASE da iteração
    config_module.r_tar_m = (1 + config_module.R_BASE)**(1/12) - 1
    
    # Aplicar parâmetros de risco da UI
    config_module.P_INFORMALIDADE = P_INFORMALIDADE_ui
    config_module.LGD = LGD_ui
    
    # Recarregar pricing_model para usar as configs atualizadas
    importlib.reload(pricing_model_module)
    from src.pricing_model import simulate_pricing

    # Parâmetros de risco da empresa (CNAE, idade, faturamento) fixos conforme UI
    rot_data_rbase = rot_df[rot_df[cnae_col] == cnae_section].iloc[0]
    p_rot_m_rbase = float(rot_data_rbase['rotatividade_mensal'])
    delay_rbase = int(rot_data_rbase['tempo_desemprego_esperado_meses'])
    sel_rbase = riscos_df[
        (riscos_df['cnae_section'] == cnae_section) &
        (riscos_df['idade_min'] <= tempo_empresa_anos) &
        (riscos_df['idade_max'] >= tempo_empresa_anos) &
        (riscos_df['faturamento_min'] <= valor_max_faturamento) &
        (riscos_df['faturamento_max'] >= valor_max_faturamento)
    ]
    p_close_annual_rbase = float(sel_rbase['prob_ajustada'].iloc[0]) if not sel_rbase.empty else 0.1002
    
    params_rbase = {
        'PV': PV_ui,
        'n': n_ui,
        'p_close_annual': p_close_annual_rbase,
        'p_rot_m': p_rot_m_rbase,
        'delay': delay_rbase
    }
    try:
        result_rbase = simulate_pricing(params_rbase)
        spread_anual_rbase.append(result_rbase['spread_anual'] * 100)
    except Exception as e:
        st.error(f"Erro ao calcular sensibilidade R_BASE para {r_base_iter*100:.0f}%: {e}")
        spread_anual_rbase.append(np.nan)

# R_BASE atual da UI para o marcador
r_base_atual_ui = (CDI_ANUAL_ui + FUNDING_spread_ui) + CUSTO_OPERACIONAL_ui + MARGEM_ui
idx_rbase = (np.abs(r_base_range - r_base_atual_ui)).argmin()
spread_atual_rbase_marker = spread_anual_rbase[idx_rbase] if 0 <= idx_rbase < len(spread_anual_rbase) else None

fig_rbase = go.Figure()
fig_rbase.add_trace(
    go.Scatter(
        x=r_base_range * 100,
        y=spread_anual_rbase,
        mode='lines+markers',
        line=dict(color='teal', width=3),
        marker=dict(size=6),
        name='Spread Anual'
    )
)

# Adiciona marcador especial para o valor selecionado na UI
if spread_atual_rbase_marker is not None and not np.isnan(spread_atual_rbase_marker):
    fig_rbase.add_trace(
        go.Scatter(
            x=[r_base_range[idx_rbase] * 100],
            y=[spread_atual_rbase_marker],
            mode='markers+text',
            marker=dict(size=16, color='crimson', symbol='diamond'),
            text=[f"{r_base_range[idx_rbase]*100:.1f}%"],
            textposition='top center',
            name='R_BASE da UI',
            showlegend=False
        )
    )

fig_rbase.update_layout(
    title="Sensibilidade do Spread Anual à Taxa Base Anual (R_BASE = Funding + Custo Op. + Margem)",
    xaxis_title="Taxa Base Anual (R_BASE) (%)",
    yaxis_title="Spread Anual (%)",
    height=400,
    margin=dict(l=60, r=40, t=60, b=40)
)
st.plotly_chart(fig_rbase, use_container_width=True)

# --- GRÁFICOS DE SENSIBILIDADE: PARÂMETROS DO EMPRÉSTIMO ---
st.subheader("📝 Sensibilidade do Spread Anual - Parâmetros do Empréstimo")

# Layout em duas colunas para os parâmetros do empréstimo
col_parcelas, col_delay = st.columns(2)

with col_parcelas:
    # Faixa de número de parcelas para simulação
    parcelas_range = np.arange(6, 61, 10) # De 6 a 60, passo de 10
    spread_anual_parcelas = []

    # Restaurar configurações financeiras e de risco para os valores da UI uma vez antes do loop
    importlib.reload(config_module)
    config_module.CDI_ANUAL = CDI_ANUAL_ui
    actual_funding_rate_parcelas = CDI_ANUAL_ui + FUNDING_spread_ui
    config_module.FUNDING = actual_funding_rate_parcelas
    config_module.CUSTO_OPERACIONAL = CUSTO_OPERACIONAL_ui
    config_module.MARGEM = MARGEM_ui
    config_module.R_BASE = config_module.FUNDING + config_module.CUSTO_OPERACIONAL + config_module.MARGEM
    config_module.r_tar_m = (1 + config_module.R_BASE)**(1/12) - 1
    config_module.P_INFORMALIDADE = P_INFORMALIDADE_ui
    config_module.LGD = LGD_ui
    importlib.reload(pricing_model_module)
    from src.pricing_model import simulate_pricing

    for n_iter in parcelas_range:
        # Parâmetros de risco da empresa (CNAE, idade, faturamento) fixos conforme UI
        rot_data_parc = rot_df[rot_df[cnae_col] == cnae_section].iloc[0]
        p_rot_m_parc = float(rot_data_parc['rotatividade_mensal'])
        delay_parc = int(rot_data_parc['tempo_desemprego_esperado_meses'])
        sel_parc = riscos_df[
            (riscos_df['cnae_section'] == cnae_section) &
            (riscos_df['idade_min'] <= tempo_empresa_anos) &
            (riscos_df['idade_max'] >= tempo_empresa_anos) &
            (riscos_df['faturamento_min'] <= valor_max_faturamento) &
            (riscos_df['faturamento_max'] >= valor_max_faturamento)
        ]
        p_close_annual_parc = float(sel_parc['prob_ajustada'].iloc[0]) if not sel_parc.empty else 0.1002
        
        params_parc = {
            'PV': PV_ui,
            'n': n_iter, # Variando o número de parcelas
            'p_close_annual': p_close_annual_parc,
            'p_rot_m': p_rot_m_parc,
            'delay': delay_parc
        }
        try:
            result_parc = simulate_pricing(params_parc)
            spread_anual_parcelas.append(result_parc['spread_anual'] * 100)
        except Exception as e:
            spread_anual_parcelas.append(np.nan)

    # Número de parcelas atual da UI para o marcador
    n_atual_ui = n_ui
    # Encontrar o índice mais próximo no range, caso o valor da UI não seja exato
    idx_parc = (np.abs(parcelas_range - n_atual_ui)).argmin()
    spread_atual_parc_marker = spread_anual_parcelas[idx_parc] if 0 <= idx_parc < len(spread_anual_parcelas) else None

    fig_parc = go.Figure()
    fig_parc.add_trace(
        go.Scatter(
            x=parcelas_range,
            y=spread_anual_parcelas,
            mode='lines+markers',
            line=dict(color='sienna', width=3),
            marker=dict(size=4),
            name='Spread Anual'
        )
    )

    # Adiciona marcador especial para o valor selecionado na UI
    if spread_atual_parc_marker is not None and not np.isnan(spread_atual_parc_marker):
        fig_parc.add_trace(
            go.Scatter(
                x=[parcelas_range[idx_parc]], # Usar o valor do range mais próximo
                y=[spread_atual_parc_marker],
                mode='markers+text',
                marker=dict(size=12, color='darkviolet', symbol='diamond'),
                text=[f"{parcelas_range[idx_parc]} parcelas"],
                textposition='top center',
                name='Nº Parcelas da UI',
                showlegend=False
            )
        )

    fig_parc.update_layout(
        title="Número de Parcelas (n)",
        xaxis_title="Número de Parcelas (n)",
        yaxis_title="Spread Anual (%)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_parc, use_container_width=True)

with col_delay:
    # Faixa de delay para simulação (0 a 12 meses)
    delay_range = np.arange(0, 13, 1)  # De 0 a 12 meses, passo de 1
    spread_anual_delay = []

    # Restaurar configurações financeiras e de risco para os valores da UI uma vez antes do loop
    importlib.reload(config_module)
    config_module.CDI_ANUAL = CDI_ANUAL_ui
    actual_funding_rate_delay = CDI_ANUAL_ui + FUNDING_spread_ui
    config_module.FUNDING = actual_funding_rate_delay
    config_module.CUSTO_OPERACIONAL = CUSTO_OPERACIONAL_ui
    config_module.MARGEM = MARGEM_ui
    config_module.R_BASE = config_module.FUNDING + config_module.CUSTO_OPERACIONAL + config_module.MARGEM
    config_module.r_tar_m = (1 + config_module.R_BASE)**(1/12) - 1
    config_module.P_INFORMALIDADE = P_INFORMALIDADE_ui
    config_module.LGD = LGD_ui
    importlib.reload(pricing_model_module)
    from src.pricing_model import simulate_pricing

    for delay_iter in delay_range:
        # Parâmetros de risco da empresa (CNAE, idade, faturamento) fixos conforme UI
        rot_data_delay = rot_df[rot_df[cnae_col] == cnae_section].iloc[0]
        p_rot_m_delay = float(rot_data_delay['rotatividade_mensal'])
        sel_delay = riscos_df[
            (riscos_df['cnae_section'] == cnae_section) &
            (riscos_df['idade_min'] <= tempo_empresa_anos) &
            (riscos_df['idade_max'] >= tempo_empresa_anos) &
            (riscos_df['faturamento_min'] <= valor_max_faturamento) &
            (riscos_df['faturamento_max'] >= valor_max_faturamento)
        ]
        p_close_annual_delay = float(sel_delay['prob_ajustada'].iloc[0]) if not sel_delay.empty else 0.1002
        
        params_delay = {
            'PV': PV_ui,
            'n': n_ui,
            'p_close_annual': p_close_annual_delay,
            'p_rot_m': p_rot_m_delay,
            'delay': delay_iter  # Variando o delay
        }
        try:
            result_delay = simulate_pricing(params_delay)
            spread_anual_delay.append(result_delay['spread_anual'] * 100)
        except Exception as e:
            spread_anual_delay.append(np.nan)

    # Valor de delay atual da UI para o marcador
    delay_atual_ui = delay  # delay vem dos dados do CNAE selecionado
    # Encontrar o índice mais próximo no range, caso o valor da UI não seja exato
    idx_delay = (np.abs(delay_range - delay_atual_ui)).argmin()
    spread_atual_delay_marker = spread_anual_delay[idx_delay] if 0 <= idx_delay < len(spread_anual_delay) else None

    fig_delay = go.Figure()
    fig_delay.add_trace(
        go.Scatter(
            x=delay_range,
            y=spread_anual_delay,
            mode='lines+markers',
            line=dict(color='chocolate', width=3),
            marker=dict(size=4),
            name='Spread Anual'
        )
    )

    # Adiciona marcador especial para o valor selecionado na UI
    if spread_atual_delay_marker is not None and not np.isnan(spread_atual_delay_marker):
        fig_delay.add_trace(
            go.Scatter(
                x=[delay_range[idx_delay]], # Usar o valor do range mais próximo
                y=[spread_atual_delay_marker],
                mode='markers+text',
                marker=dict(size=12, color='darkorange', symbol='diamond'),
                text=[f"{delay_range[idx_delay]} meses"],
                textposition='top center',
                name='Delay da UI',
                showlegend=False
            )
        )

    fig_delay.update_layout(
        title="Delay (Tempo de Desemprego)",
        xaxis_title="Delay (meses)",
        yaxis_title="Spread Anual (%)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_delay, use_container_width=True)

# --- GRÁFICOS DE SENSIBILIDADE: CARACTERÍSTICAS DO TOMADOR ---
st.subheader("👤 Sensibilidade do Spread Anual - Características do Tomador")

# Layout em duas colunas para as características do tomador
col_idade_tomador, col_genero = st.columns(2)

with col_idade_tomador:
    # Faixa de idades para simulação
    idade_tomador_range = np.arange(21, 76, 5)  # De 21 a 75 anos, passo de 5
    spread_anual_idade_tomador = []

    # Restaurar configurações financeiras e de risco para os valores da UI uma vez antes do loop
    importlib.reload(config_module)
    config_module.CDI_ANUAL = CDI_ANUAL_ui
    actual_funding_rate_idade_tomador = CDI_ANUAL_ui + FUNDING_spread_ui
    config_module.FUNDING = actual_funding_rate_idade_tomador
    config_module.CUSTO_OPERACIONAL = CUSTO_OPERACIONAL_ui
    config_module.MARGEM = MARGEM_ui
    config_module.R_BASE = config_module.FUNDING + config_module.CUSTO_OPERACIONAL + config_module.MARGEM
    config_module.r_tar_m = (1 + config_module.R_BASE)**(1/12) - 1
    config_module.P_INFORMALIDADE = P_INFORMALIDADE_ui
    config_module.LGD = LGD_ui
    importlib.reload(pricing_model_module)
    from src.pricing_model import simulate_pricing

    for idade_iter in idade_tomador_range:
        # Buscar probabilidade de óbito para a idade iterada
        obito_sel_iter = obito_df[
            (obito_df['gender'] == sexo_ui) &
            (obito_df['idade_anos'] == idade_iter)
        ]
        p_obito_ann_iter = float(obito_sel_iter['prob_anual_obito'].iloc[0]) if not obito_sel_iter.empty else 0.0
        
        # Parâmetros de risco da empresa fixos conforme UI
        rot_data_idade_tomador = rot_df[rot_df[cnae_col] == cnae_section].iloc[0]
        p_rot_m_idade_tomador = float(rot_data_idade_tomador['rotatividade_mensal'])
        delay_idade_tomador = int(rot_data_idade_tomador['tempo_desemprego_esperado_meses'])
        sel_idade_tomador = riscos_df[
            (riscos_df['cnae_section'] == cnae_section) &
            (riscos_df['idade_min'] <= tempo_empresa_anos) &
            (riscos_df['idade_max'] >= tempo_empresa_anos) &
            (riscos_df['faturamento_min'] <= valor_max_faturamento) &
            (riscos_df['faturamento_max'] >= valor_max_faturamento)
        ]
        p_close_annual_idade_tomador = float(sel_idade_tomador['prob_ajustada'].iloc[0]) if not sel_idade_tomador.empty else 0.1002
        
        params_idade_tomador = {
            'PV': PV_ui,
            'n': n_ui,
            'p_close_annual': p_close_annual_idade_tomador,
            'p_rot_m': p_rot_m_idade_tomador,
            'delay': delay_idade_tomador,
            'p_obito_ann': p_obito_ann_iter  # Variando a probabilidade de óbito
        }
        try:
            result_idade_tomador = simulate_pricing(params_idade_tomador)
            spread_anual_idade_tomador.append(result_idade_tomador['spread_anual'] * 100)
        except Exception as e:
            spread_anual_idade_tomador.append(np.nan)

    # Valor de idade atual da UI para o marcador
    idade_atual_tomador = idade_ui
    # Encontrar o índice mais próximo no range
    idx_idade_tomador = (np.abs(idade_tomador_range - idade_atual_tomador)).argmin()
    spread_atual_idade_tomador_marker = spread_anual_idade_tomador[idx_idade_tomador] if 0 <= idx_idade_tomador < len(spread_anual_idade_tomador) else None

    fig_idade_tomador = go.Figure()
    fig_idade_tomador.add_trace(
        go.Scatter(
            x=idade_tomador_range,
            y=spread_anual_idade_tomador,
            mode='lines+markers',
            line=dict(color='mediumorchid', width=3),
            marker=dict(size=4),
            name='Spread Anual'
        )
    )

    # Adiciona marcador especial para o valor selecionado na UI
    if spread_atual_idade_tomador_marker is not None and not np.isnan(spread_atual_idade_tomador_marker):
        fig_idade_tomador.add_trace(
            go.Scatter(
                x=[idade_tomador_range[idx_idade_tomador]],
                y=[spread_atual_idade_tomador_marker],
                mode='markers+text',
                marker=dict(size=12, color='darkred', symbol='diamond'),
                text=[f"{idade_tomador_range[idx_idade_tomador]} anos"],
                textposition='top center',
                name='Idade da UI',
                showlegend=False
            )
        )

    fig_idade_tomador.update_layout(
        title="Idade do Tomador",
        xaxis_title="Idade do Tomador (anos)",
        yaxis_title="Spread Anual (%)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_idade_tomador, use_container_width=True)

with col_genero:
    # Comparação entre gêneros
    generos = ['M', 'F']
    spread_anual_genero = []

    # Restaurar configurações financeiras e de risco para os valores da UI uma vez antes do loop
    importlib.reload(config_module)
    config_module.CDI_ANUAL = CDI_ANUAL_ui
    actual_funding_rate_genero = CDI_ANUAL_ui + FUNDING_spread_ui
    config_module.FUNDING = actual_funding_rate_genero
    config_module.CUSTO_OPERACIONAL = CUSTO_OPERACIONAL_ui
    config_module.MARGEM = MARGEM_ui
    config_module.R_BASE = config_module.FUNDING + config_module.CUSTO_OPERACIONAL + config_module.MARGEM
    config_module.r_tar_m = (1 + config_module.R_BASE)**(1/12) - 1
    config_module.P_INFORMALIDADE = P_INFORMALIDADE_ui
    config_module.LGD = LGD_ui
    importlib.reload(pricing_model_module)
    from src.pricing_model import simulate_pricing

    for genero_iter in generos:
        # Buscar probabilidade de óbito para o gênero iterado
        obito_sel_genero = obito_df[
            (obito_df['gender'] == genero_iter) &
            (obito_df['idade_anos'] == idade_ui)
        ]
        p_obito_ann_genero = float(obito_sel_genero['prob_anual_obito'].iloc[0]) if not obito_sel_genero.empty else 0.0
        
        # Parâmetros de risco da empresa fixos conforme UI
        rot_data_genero = rot_df[rot_df[cnae_col] == cnae_section].iloc[0]
        p_rot_m_genero = float(rot_data_genero['rotatividade_mensal'])
        delay_genero = int(rot_data_genero['tempo_desemprego_esperado_meses'])
        sel_genero = riscos_df[
            (riscos_df['cnae_section'] == cnae_section) &
            (riscos_df['idade_min'] <= tempo_empresa_anos) &
            (riscos_df['idade_max'] >= tempo_empresa_anos) &
            (riscos_df['faturamento_min'] <= valor_max_faturamento) &
            (riscos_df['faturamento_max'] >= valor_max_faturamento)
        ]
        p_close_annual_genero = float(sel_genero['prob_ajustada'].iloc[0]) if not sel_genero.empty else 0.1002
        
        params_genero = {
            'PV': PV_ui,
            'n': n_ui,
            'p_close_annual': p_close_annual_genero,
            'p_rot_m': p_rot_m_genero,
            'delay': delay_genero,
            'p_obito_ann': p_obito_ann_genero  # Variando a probabilidade de óbito por gênero
        }
        try:
            result_genero = simulate_pricing(params_genero)
            spread_anual_genero.append(result_genero['spread_anual'] * 100)
        except Exception as e:
            spread_anual_genero.append(np.nan)

    # Criar labels mais amigáveis
    genero_labels = ['Masculino', 'Feminino']

    fig_genero = go.Figure()
    fig_genero.add_trace(
        go.Bar(
            x=genero_labels,
            y=spread_anual_genero,
            marker_color=['lightblue', 'lightpink'],
            text=[f"{v:.3f}%" if not np.isnan(v) else "" for v in spread_anual_genero],
            textposition='auto',
            name='Spread Anual'
        )
    )

    # Destacar o gênero selecionado
    genero_atual_idx = 0 if sexo_ui == 'M' else 1
    if genero_atual_idx < len(spread_anual_genero) and not np.isnan(spread_anual_genero[genero_atual_idx]):
        fig_genero.add_trace(
            go.Scatter(
                x=[genero_labels[genero_atual_idx]],
                y=[spread_anual_genero[genero_atual_idx]],
                mode='markers+text',
                marker=dict(size=16, color='darkred', symbol='diamond'),
                text=[f"Selecionado"],
                textposition='top center',
                name='Gênero da UI',
                showlegend=False
            )
        )

    fig_genero.update_layout(
        title="Gênero do Tomador",
        xaxis_title="Gênero",
        yaxis_title="Spread Anual (%)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_genero, use_container_width=True)

# --- GRÁFICO DE TORNADO: IMPACTO RELATIVO DAS VARIÁVEIS NO SPREAD ANUAL ---
st.subheader("🌪️ Tornado – Impacto Relativo das Variáveis no Spread Anual (±10%)")

# 1) Spread base já calculado
spread_base_pp = result['spread_anual'] * 100  # pontos percentuais

# 2) Lista de variáveis a testar
variables_tornado = [
    {
        'label': 'R_BASE',
        'type': 'config',          # indica que altera config_module.R_BASE
        'get_base': lambda: (CDI_ANUAL_ui + FUNDING_spread_ui) + CUSTO_OPERACIONAL_ui + MARGEM_ui,
        'apply': lambda val: setattr(config_module, 'R_BASE', val) or setattr(config_module, 'r_tar_m', (1+val)**(1/12)-1)
    },
    {
        'label': 'LGD',
        'type': 'config',
        'get_base': lambda: LGD_ui,
        'apply': lambda val: setattr(config_module, 'LGD', val)
    },
    {
        'label': 'Informalidade',
        'type': 'config',
        'get_base': lambda: P_INFORMALIDADE_ui,
        'apply': lambda val: setattr(config_module, 'P_INFORMALIDADE', val)
    },
    {
        'label': 'Número de Parcelas (n)',
        'type': 'param',
        'get_base': lambda: n_ui,
        'apply': None  # manipulado via params
    },
    {
        'label': 'Delay',
        'type': 'param',
        'get_base': lambda: delay,
        'apply': None  # manipulado via params
    },
    {
        'label': 'Idade do Tomador',
        'type': 'param',
        'get_base': lambda: idade_ui,
        'apply': None  # manipulado via params
    }
]

bars_data = []

for var in variables_tornado:
    base_val = var['get_base']()
    # ±10% (garantindo limites plausíveis)
    low_val = base_val * 0.9
    high_val = base_val * 1.1

    # Se a variável for inteira (n, delay ou idade), arredondar
    if var['label'].startswith('Número') or var['label'] == 'Delay' or var['label'] == 'Idade do Tomador':
        # Para Número de Parcelas, use um choque fixo de ±12 para dar visibilidade
        if var['label'].startswith('Número'):
            low_val = max(int(round(base_val)) - 12, 6)
            high_val = min(int(round(base_val)) + 12, 120)
        elif var['label'] == 'Idade do Tomador':
            # Para Idade do Tomador, use um choque fixo de ±10 anos
            low_val = max(int(round(base_val)) - 10, 21)
            high_val = min(int(round(base_val)) + 10, 75)
        else:  # Delay => ±10% padrão
            low_val = int(round(low_val))
            high_val = int(round(high_val))

        # Garantir valores distintos do base
        base_int = int(round(base_val))
        if low_val == base_int:
            low_val = max(base_int - 1, 21 if var['label'] == 'Idade do Tomador' else 0)
        if high_val == base_int:
            high_val = min(base_int + 1, 75 if var['label'] == 'Idade do Tomador' else 999)
        if low_val == high_val:
            high_val = low_val + 1

    # Spreads para low / high
    diffs = {}
    for key, val in [('low', low_val), ('high', high_val)]:
        # 2.a) Resetar configurações para baseline em cada iteração
        importlib.reload(config_module)
        # Reaplicar valores da UI
        config_module.CDI_ANUAL = CDI_ANUAL_ui
        config_module.FUNDING = CDI_ANUAL_ui + FUNDING_spread_ui
        config_module.CUSTO_OPERACIONAL = CUSTO_OPERACIONAL_ui
        config_module.MARGEM = MARGEM_ui
        config_module.R_BASE = config_module.FUNDING + config_module.CUSTO_OPERACIONAL + config_module.MARGEM
        config_module.r_tar_m = (1 + config_module.R_BASE)**(1/12) - 1
        config_module.P_INFORMALIDADE = P_INFORMALIDADE_ui
        config_module.LGD = LGD_ui

        # Aplicar choque específico
        if var['type'] == 'config':
            var['apply'](val)

        # Buscar probabilidade de óbito para a idade (pode ser variada)
        idade_para_obito = idade_ui if var['label'] != 'Idade do Tomador' else int(val)
        obito_sel_tornado = obito_df[
            (obito_df['gender'] == sexo_ui) &
            (obito_df['idade_anos'] == idade_para_obito)
        ]
        p_obito_ann_tornado = float(obito_sel_tornado['prob_anual_obito'].iloc[0]) if not obito_sel_tornado.empty else 0.0
        
        # Preparar parâmetros do empréstimo
        params_var = {
            'PV': PV_ui,
            'n': n_ui if var['label'] != 'Número de Parcelas (n)' else int(val),
            'p_close_annual': p_close_annual,
            'p_rot_m': p_rot_m,
            'delay': delay if var['label'] != 'Delay' else int(val),
            'p_obito_ann': p_obito_ann_tornado
        }

        # Recarregar pricing_model após alterações no config
        importlib.reload(pricing_model_module)
        from src.pricing_model import simulate_pricing
        try:
            res_var = simulate_pricing(params_var)
            spread_var = res_var['spread_anual'] * 100
            diffs[key] = spread_var - spread_base_pp
        except Exception as exc:
            st.error(f"Erro ao calcular tornado para {var['label']} {key}: {exc}")
            diffs[key] = np.nan

    bars_data.append({
        'label': var['label'] + f"\n({base_val:.4g} → {low_val:.4g}/{high_val:.4g})",
        'low_diff': diffs['low'],
        'high_diff': diffs['high']
    })

# Ordenar por impacto máximo absoluto
a_impact = lambda b: max(abs(b['low_diff']), abs(b['high_diff']))
bars_data.sort(key=a_impact, reverse=True)

labels_tornado = [b['label'] for b in bars_data]
low_vals = [b['low_diff'] for b in bars_data]
high_vals = [b['high_diff'] for b in bars_data]

fig_tornado = go.Figure()
fig_tornado.add_trace(
    go.Bar(
        x=low_vals,
        y=labels_tornado,
        orientation='h',
        marker_color='indianred',
        name='-10%'
    )
)
fig_tornado.add_trace(
    go.Bar(
        x=high_vals,
        y=labels_tornado,
        orientation='h',
        marker_color='seagreen',
        name='+10%'
    )
)

# Destaque visual de importância pelo espessamento das barras (largura)
fig_tornado.update_traces(marker_line_width=1.5)

fig_tornado.update_layout(
    barmode='overlay',
    title='Tornado – Variação do Spread Anual para ±10% em Cada Variável',
    xaxis_title='Δ Spread Anual (p.p.)',
    yaxis_title='',
    height=400,
    bargap=0.15,
    legend_title='Choque',
    margin=dict(l=200, r=40, t=60, b=40)
)

st.plotly_chart(fig_tornado, use_container_width=True)

# --- CURVAS DE RISCO AO LONGO DO TEMPO ---
st.subheader("⏳ Curvas de Risco ao Longo do Tempo")

# Recalcular curvas S, f_def, f_del, f_death localmente usando os resultados da simulação principal
n_sim = result.get('n')
h_default_sim = result.get('h_default')
h_delay_sim = result.get('h_delay')
h_death_sim = result.get('h_death')

if n_sim is not None and h_default_sim is not None and h_delay_sim is not None and h_death_sim is not None:
    S_curve_sim = np.empty(n_sim + 1)
    S_curve_sim[0] = 1.0
    f_def_curve_sim = np.zeros(n_sim + 1)
    f_del_curve_sim = np.zeros(n_sim + 1)
    f_death_curve_sim = np.zeros(n_sim + 1)

    for u in range(1, n_sim + 1):
        S_curve_sim[u] = S_curve_sim[u-1] * (1 - h_default_sim - h_delay_sim - h_death_sim)
        f_def_curve_sim[u] = S_curve_sim[u-1] * h_default_sim
        f_del_curve_sim[u] = S_curve_sim[u-1] * h_delay_sim
        f_death_curve_sim[u] = S_curve_sim[u-1] * h_death_sim
    
    t_range = np.arange(n_sim + 1) # Array de meses [0, 1, ..., n]

    fig_risk_curves = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        subplot_titles=(
            'Probabilidade de Sobrevivência Acumulada (S[t])',
            'Probabilidade de Default no Mês t (f_def[t])',
            'Probabilidade de Atraso no Mês t (f_del[t])',
            'Probabilidade de Óbito no Mês t (f_death[t])'
        ),
        vertical_spacing=0.08
    )

    # Curva S[t]
    fig_risk_curves.add_trace(
        go.Scatter(x=t_range[1:], y=S_curve_sim[1:], name='S(t)', mode='lines', line=dict(color='blue')),
        row=1, col=1
    )
    # Curva f_def[t]
    fig_risk_curves.add_trace(
        go.Scatter(x=t_range[1:], y=f_def_curve_sim[1:], name='f_def(t)', mode='lines', line=dict(color='red')),
        row=2, col=1
    )
    # Curva f_del[t]
    fig_risk_curves.add_trace(
        go.Scatter(x=t_range[1:], y=f_del_curve_sim[1:], name='f_del(t)', mode='lines', line=dict(color='orange')),
        row=3, col=1
    )
    # Curva f_death[t]
    fig_risk_curves.add_trace(
        go.Scatter(x=t_range[1:], y=f_death_curve_sim[1:], name='f_death(t)', mode='lines', line=dict(color='purple')),
        row=4, col=1
    )

    fig_risk_curves.update_layout(
        height=600, 
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_risk_curves.update_xaxes(title_text="Meses do Empréstimo (t)", row=4, col=1)
    fig_risk_curves.update_yaxes(title_text="Probabilidade", row=1, col=1, tickformat=".2%")
    fig_risk_curves.update_yaxes(title_text="Probabilidade", row=2, col=1, tickformat=".2%")
    fig_risk_curves.update_yaxes(title_text="Probabilidade", row=3, col=1, tickformat=".2%")
    fig_risk_curves.update_yaxes(title_text="Probabilidade", row=4, col=1, tickformat=".2%")

    st.plotly_chart(fig_risk_curves, use_container_width=True)
else:
    st.info("Não foi possível gerar as curvas de risco. Verifique os parâmetros da simulação e os resultados retornados.")

# --- FÓRMULAS MATEMÁTICAS DO MODELO DE PRICING ---
st.subheader("📐 Metodologia de Cálculo do Pricing Ajustado ao Risco")

with st.expander("Ver fórmulas matemáticas e passos de cálculo"):
    st.markdown(r"""
    ### Passo a Passo do Cálculo do Pricing

    #### 1) Conversão de Probabilidades Anuais em Hazards Mensais

    * **Hazard de fechamento (mensal)**: 
      $h_{close} = 1 - (1 - p_{close\_ann})^{1/12}$
    
    * **Hazard de turnover total (incluindo fechamento)**:
      $h_{turnover} = h_{close} + (1 - h_{close}) \times (p_{rot\_m} \times 0.66)$
    
    * **Hazard de default (parte do turnover que vira inadimplência)**:
      $h_{default} = h_{turnover} \times P_{INFORMALIDADE}$
    
    * **Hazard de atraso (parte do turnover que apenas atrasa)**:
      $h_{delay} = h_{turnover} \times (1 - P_{INFORMALIDADE})$
    
    * **Hazard de óbito (mensal)**:
      $h_{death} = 1 - (1 - p_{obito\_ann})^{1/12}$

    #### 2) Construção da Curva de Sobrevivência e Densidades

    Para cada mês $u$ do $1$ até $n$:
    
    * **Probabilidade de sobrevivência até o mês $u$**:
      $S[u] = S[u-1] \times (1 - h_{default} - h_{delay} - h_{death})$, com $S[0] = 1.0$
    
    * **Densidade de probabilidade de default no mês $u$**:
      $f_{def}[u] = S[u-1] \times h_{default}$
    
    * **Densidade de probabilidade de atraso no mês $u$**:
      $f_{del}[u] = S[u-1] \times h_{delay}$
    
    * **Densidade de probabilidade de óbito no mês $u$**:
      $f_{death}[u] = S[u-1] \times h_{death}$

    #### 3) Cálculo do PMT Base (Livre de Risco)

    * **Taxa-meta mensal** (derivada da taxa base anual):
      $r_{tar\_m} = (1 + R_{BASE})^{1/12} - 1$
    
    * **PMT Base** (parcela livre de risco):
      $PMT_{base} = PV \times \frac{r_{tar\_m}}{1 - (1 + r_{tar\_m})^{-n}}$

    #### 4) Cálculo do PMT Ajustado ao Risco (PMT_risco)

    Queremos encontrar o valor da parcela mensal $PMT_{risco}$ tal que o valor presente esperado dos fluxos futuros $(EPV_{total})$ iguale o valor do empréstimo $(PV)$:

    $EPV_{total} = EPV_{surv} + EPV_{delay} + EPV_{default} + EPV_{death} = PV$

    **4.1. EPV da sobrevivência até o fim (sem evento)**:

    $EPV_{surv} = S[n] \times \sum_{t=1}^{n} \frac{PMT_{risco}}{(1 + r_{tar\_m})^t}$

    **4.2. EPV em caso de atraso (evento ocorre no mês $u \leq n$)**:

    $EPV_{delay} = \sum_{u=1}^{n} f_{del}[u] \times \left[ \sum_{t=1}^{u-1} \frac{PMT_{risco}}{(1 + r_{tar\_m})^t} + \sum_{t=u}^{n} \frac{PMT_{risco}}{(1 + r_{tar\_m})^{t + d}} \right]$

    onde $d$ é o número de meses de atraso após o evento de desligamento.

    **4.3. EPV em caso de default (evento ocorre no mês $u \leq n$)**:

    $EPV_{default} = \sum_{u=1}^{n} f_{def}[u] \times \left[ \sum_{t=1}^{u-1} \frac{PMT_{risco}}{(1 + r_{tar\_m})^t} + \frac{(1 - LGD) \times saldo[u-1]}{(1 + r_{tar\_m})^u} \right]$

    onde $saldo[u-1]$ é o saldo devedor imediatamente antes do default.

    **4.4. EPV em caso de óbito (evento ocorre no mês $u \leq n$)**:

    $EPV_{death} = \sum_{u=1}^{n} f_{death}[u] \times \left[ \sum_{t=1}^{u-1} \frac{PMT_{risco}}{(1 + r_{tar\_m})^t} + \frac{(1 - LGD_{DEATH}) \times saldo[u-1]}{(1 + r_{tar\_m})^u} \right]$

    onde $LGD_{DEATH}$ é a perda dada o óbito e $saldo[u-1]$ é o saldo devedor imediatamente antes do óbito.

    **4.5. Resolução numérica**:

    A equação:
    $EPV_{surv} + EPV_{delay} + EPV_{default} + EPV_{death} = PV$

    é resolvida numericamente com `fsolve`, ajustando $PMT_{risco}$ até que a igualdade se satisfaça.

    #### 5) Spreads e TIR do Cliente

    * **Spread em valor**: 
      $spread_{valor} = PMT_{risco} - PMT_{base}$
    
    * **Taxa mínima mensal (TIR mensal)** é calculada resolvendo:
      $\sum_{t=1}^{n} \frac{PMT_{risco}}{(1 + r_{min\_m})^t} = PV$

    * **Taxa mínima anual**: 
      $R_{min\_anual} = (1 + r_{min\_m})^{12} - 1$
    
    * **Spread (em pontos base)**: 
      $spread = r_{min\_m} - r_{tar\_m}$
    
    * **Spread anual (em pontos base)**: 
      $spread_{anual} = R_{min\_anual} - R_{BASE}$

    #### 6) KPIs Adicionais

    * **Probabilidade de default total**: 
      $P_{default\_total} = \sum_{u=1}^{n} f_{def}[u]$
    
    * **Probabilidade de atraso total**: 
      $P_{delay\_total} = \sum_{u=1}^{n} f_{del}[u]$
    
    * **Probabilidade de óbito total**: 
      $P_{death\_total} = \sum_{u=1}^{n} f_{death}[u]$
    
    * **Parcelas esperadas** (pagas no prazo): 
      $Expected\_payments = \sum_{u=0}^{n-1} S[u]$
    
    * **Tempo médio até evento**: 
      $Mean\_time\_to\_event = \sum_{u=1}^{n} u \times (f_{def}[u] + f_{del}[u] + f_{death}[u])$
    
    * **Duração esperada**: 
      $E\_Duration = \sum_{u=1}^{n} u \times f_{def}[u] + \sum_{u=1}^{n} (n + d) \times f_{del}[u] + \sum_{u=1}^{n} u \times f_{death}[u] + n \times S[n]$
    
    * **LGD ponderado**: 
      $LGD_{ponderado} = \frac{\sum_{u=1}^{n} f_{def}[u] \times LGD \times saldo[u-1]}{PV}$
    
    * **LGD ponderado óbito**: 
      $LGD_{ponderado\_death} = \frac{\sum_{u=1}^{n} f_{death}[u] \times LGD_{DEATH} \times saldo[u-1]}{PV}$
    """)
