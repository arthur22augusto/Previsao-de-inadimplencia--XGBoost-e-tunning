import os
import pandas as pd
import numpy as np

def baixar_dados():
    # Base URL dos arquivos
    base_url = "https://raw.githubusercontent.com/datarisk-io/datarisk-case-ds-junior/177db4cc1cc2dc883e1f7f1cb205ceda365fe51d/data/"

    # Nomes dos arquivos
    file_names = {
        "base_cadastral": "base_cadastral.csv",
        "base_info": "base_info.csv",
        "base_pagamentos_desenvolvimento": "base_pagamentos_desenvolvimento.csv",
        "base_pagamentos_teste": "base_pagamentos_teste.csv"
    }

    # Caminho para a pasta raw
    raw_data_path = "./data/raw/"
    os.makedirs(raw_data_path, exist_ok=True)  # Garante que a pasta existe

    for file_name in file_names.values():
        file_url = base_url + file_name
        local_file_path = os.path.join(raw_data_path, file_name)

        try:
            # Ler e salvar o arquivo
            df = pd.read_csv(file_url, sep=";")
            df.to_csv(local_file_path, sep=";", index=False)
            print(f"✅ {file_name} salvo em {local_file_path}")
        except Exception as e:
            print(f"❌ Erro ao processar {file_name}: {e}")



def carregar_dados():
    # Caminho base dos arquivos salvos
    raw_data_path = "data/raw/"

    # Nomes dos arquivos (devem coincidir com os da função de download)
    file_names = {
        "base_cadastral": "base_cadastral.csv",
        "base_info": "base_info.csv",
        "base_pagamentos_desenvolvimento": "base_pagamentos_desenvolvimento.csv",
        "base_pagamentos_teste": "base_pagamentos_teste.csv"
    }

    # Ler os arquivos locais em dataframes
    dataframes = {}
    for df_name, file_name in file_names.items():
        local_file_path = os.path.join(raw_data_path, file_name)
        try:
            dataframes[df_name] = pd.read_csv(local_file_path, sep=";")
        except Exception as e:
            print(f"❌ Erro ao carregar {file_name}: {e}")

    # Atribuir os DataFrames às variáveis principais
    data_cadastro = dataframes.get("base_cadastral")
    data_hist = dataframes.get("base_info")
    data_pagamentos = dataframes.get("base_pagamentos_desenvolvimento")
    data_objetivo = dataframes.get("base_pagamentos_teste")

    return data_cadastro, data_hist, data_pagamentos, data_objetivo
