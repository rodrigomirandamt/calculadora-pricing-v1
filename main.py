#!/usr/bin/env python
# main.py
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
"""
import argparse
import pandas as pd
import os
import sys
import datetime
import inspect
import concurrent.futures
import multiprocessing
from pathlib import Path
from itertools import islice
from functools import partial
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Imports dos módulos da aplicação
from src.config import *
from src.data_utils import load_data, process_row
from src.validacao import validate_results
# Importar simulate_pricing aqui para evitar problemas de pickling com ProcessPoolExecutor
from src.pricing_model import simulate_pricing

def generate_report(output_csv_path, args, base_path, rot_path, risk_path, obito_path):
    """
    Gera um relatório com metadados e validação dos resultados
    
    Args:
        output_csv_path: Caminho do arquivo CSV de resultados
        args: Argumentos da linha de comando
        base_path: Caminho efetivamente usado para o arquivo base
        rot_path: Caminho efetivamente usado para o arquivo de rotatividade
        risk_path: Caminho efetivamente usado para o arquivo de riscos
        obito_path: Caminho efetivamente usado para o arquivo de óbito
        
    Returns:
        str: Caminho do arquivo de relatório gerado
    """
    today = datetime.datetime.now()
    report_filename = os.path.splitext(output_csv_path)[0] + '.txt'
    
    # Extrai informações dos módulos
    module_info = {
        'config': inspect.getmodule(VERSION),
        'data_utils': inspect.getmodule(load_data),
        'pricing_model': inspect.getmodule(simulate_pricing)
    }
    
    # Coleta hipóteses de config
    config_params = {
        'CDI_ANUAL': CDI_ANUAL,
        'FUNDING': FUNDING,
        'CUSTO_OPERACIONAL': CUSTO_OPERACIONAL,
        'MARGEM': MARGEM,
        'R_BASE': R_BASE,
        'P_INFORMALIDADE': P_INFORMALIDADE,
        'LGD': LGD,
        'LGD_DEATH': LGD_DEATH
    }
    
    # Arquivos utilizados
    files_used = {
        'BASE': base_path,
        'ROTATIVIDADE': rot_path,
        'FECHAMENTO': risk_path,
        'OBITO': obito_path
    }
    
    # Gera o conteúdo do relatório
    report = []
    report.append("=" * 80)
    report.append(f"RELATÓRIO DE PROCESSAMENTO - CALCULADORA PRICING v{VERSION}")
    report.append("=" * 80)
    report.append(f"Data e hora: {today.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Arquivo processado: {output_csv_path}")
    report.append("")
    
    # Adiciona informações da linha de comando
    report.append("-" * 80)
    report.append("PARÂMETROS DE EXECUÇÃO")
    report.append("-" * 80)
    if args.num:
        report.append(f"Número de linhas processadas: {args.num}")
    if args.sample:
        report.append(f"Tamanho da amostra: {args.sample}")
    report.append("")
    
    # Adiciona informações dos arquivos
    report.append("-" * 80)
    report.append("ARQUIVOS UTILIZADOS")
    report.append("-" * 80)
    for name, path in files_used.items():
        report.append(f"{name}: {path}")
    report.append("")
    
    # Adiciona hipóteses
    report.append("-" * 80)
    report.append("HIPÓTESES UTILIZADAS")
    report.append("-" * 80)
    for name, value in config_params.items():
        report.append(f"{name}: {value}")
    report.append("")
    
    # Adiciona resultados da validação
    report.append("-" * 80)
    report.append("RESULTADOS DA VALIDAÇÃO")
    report.append("-" * 80)
    
    try:
        # Usar diretamente a função de validação
        validation_results = validate_results(output_csv_path)
        report.append(validation_results)
    except Exception as e:
        report.append(f"Erro ao executar validação: {e}")
    
    # Salva o relatório
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return report_filename

def process_batch(batch, rot_df, cnae_col, riscos_df):
    """
    Processa um lote de registros em paralelo
    
    Args:
        batch: Lista de registros (dicionários) a serem processados
        rot_df: DataFrame de rotatividade
        cnae_col: Nome da coluna de CNAE
        riscos_df: DataFrame de riscos de fechamento
        
    Returns:
        list: Lista de resultados processados
    """
    return [process_row(row, rot_df, cnae_col, riscos_df) for row in batch]

def chunk_list(lst, n):
    """Divide uma lista em pedaços de tamanho n"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, help='Número de linhas a processar')
    parser.add_argument('--sample', type=int, help='Tamanho da amostra aleatória')
    parser.add_argument('-o', '--output', help='Arquivo de saída (opcional)')
    parser.add_argument('--base-path', help='Caminho para o arquivo base (opcional)')
    parser.add_argument('--rot-path', help='Caminho para o arquivo de rotatividade (opcional)')
    parser.add_argument('--risk-path', help='Caminho para o arquivo de riscos (opcional)')
    parser.add_argument('--obito-path', help='Caminho para o arquivo de óbito (opcional)')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help='Número de workers para processamento paralelo')
    parser.add_argument('--batch-size', type=int, default=200, help='Tamanho do lote para processamento')
    args = parser.parse_args()
    
    # Define nome do arquivo com data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE}_{today}.csv")

    # Resolve caminhos dos arquivos de entrada
    base_path = args.base_path or CSV_BASE
    rot_path = args.rot_path or CSV_ROTATIVIDADE
    risk_path = args.risk_path or CSV_FECHAMENTO
    obito_path = getattr(args, 'obito_path', None) or CSV_OBITO

    # Load data
    base, rot_df, riscos_df, cnae_col = load_data(
        base_path=base_path,
        rot_path=rot_path,
        risk_path=risk_path,
        obito_path=obito_path,
        num_rows=args.num, 
        sample_size=args.sample
    )
    records = base.to_dict('records')
    
    # Definir número de workers e tamanho do lote
    num_workers = args.workers
    batch_size = args.batch_size
    
    # Dividir os registros em lotes
    batches = list(chunk_list(records, batch_size))
    
    # Process contracts using concurrent.futures
    results = []
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
                 TimeElapsedColumn(), TimeRemainingColumn()) as prog:
        task = prog.add_task("Calculando pricing...", total=len(batches))
        
        # Criar uma função parcial com os parâmetros fixos
        process_batch_partial = partial(process_batch, rot_df=rot_df, cnae_col=cnae_col, riscos_df=riscos_df)
        
        # Garantir que multiprocessing use o método 'spawn' em Windows para evitar problemas
        if os.name == 'nt':  # Windows
            ctx = multiprocessing.get_context('spawn')
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
        else:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
        
        # Processar os lotes em paralelo com ProcessPoolExecutor para utilizar múltiplos CPUs
        with executor:
            # Submeter os lotes para processamento
            future_to_batch = {executor.submit(process_batch_partial, batch): i for i, batch in enumerate(batches)}
            
            # Coletar os resultados à medida que são concluídos
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as exc:
                    print(f"Um lote gerou uma exceção: {exc}")
                finally:
                    prog.update(task, advance=1)

    # Create and export dataframe
    df = pd.DataFrame(results)

    # Order columns as specified
    cols_order = [
        'personid', 'contractid', 'cnae_section', 'porte',
        'valor_max_faturamento', 'tempo_empresa_anos', 'cluster_person',
        'idade', 'sexo',
        'PV', 'n',
        'CDI_anual', 'Funding', 'Custo_Operacional', 'Margem', 'R_base_anual',
        'delay', 'p_rot_m', 'p_close_annual', 'p_obito_ann',
        'h_close', 'h_turnover', 'h_default', 'h_delay', 'h_death',
        'S_n', 'P_default_total', 'P_delay_total', 'P_death_total',
        'EPV_surv', 'EPV_delay', 'EPV_default', 'EPV_death',
        'Expected_payments', 'Mean_time_to_event', 'E_Duration', 'LGD_ponderado', 'LGD_ponderado_death',
        'PMT_base', 'PMT_risco', 'spread_valor',
        'r_tar_m', 'r_min_m', 'spread',
        'R_min_anual', 'spread_anual', 'calc_time_s'
    ]
    df = df[cols_order]

    # Garantir que a pasta de resultados exista
    output_path = Path(output_file)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Save results
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ Resultados salvos em {output_file}")
    
    # Gerar relatório de acompanhamento
    report_file = generate_report(output_file, args, base_path, rot_path, risk_path, obito_path)
    print(f"✅ Relatório gerado em {report_file}")

if __name__ == '__main__':
    main() 