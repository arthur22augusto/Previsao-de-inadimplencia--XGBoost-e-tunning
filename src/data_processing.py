import pandas as pd
import numpy as np

# Função para converter os tipos de dados
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        c for c in df.columns
        if c.startswith("DATA_") or c.startswith("SAFRA_REF")
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# Função para ajustar o flag_pf
def adjust_flag_pf(df: pd.DataFrame) -> pd.DataFrame:
    if "FLAG_PF" not in df.columns:
        return df
    df = df.copy()
    df["FLAG_PF"] = df["FLAG_PF"].fillna("PJ").replace({"X": "PF"})
    df["FLAG_PF"] = (df["FLAG_PF"] == "PF").astype(int)
    return df

# Função para ajustar o ddd
def adjust_ddd(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["DDD"], errors="ignore")

# Função para preencher os valores nulos da taxa e valor
def fill_null_values_taxa_valor(df: pd.DataFrame) -> pd.DataFrame:
    mean_by_cli_taxa = df.groupby("ID_CLIENTE")["TAXA"].transform("mean")
    df["TAXA"] = df["TAXA"].fillna(mean_by_cli_taxa).fillna(df["TAXA"].mean())

    mean_by_cli_val = df.groupby("ID_CLIENTE")["VALOR_A_PAGAR"].transform("mean")
    med_val = df["VALOR_A_PAGAR"].median()
    df["VALOR_A_PAGAR"] = df["VALOR_A_PAGAR"].fillna(mean_by_cli_val).fillna(med_val)
    return df


# Função para mesclar os dados iniciais
def merge_data(
    df_pagamentos: pd.DataFrame,
    df_cadastro: pd.DataFrame,
    df_hist: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df_pagamentos.merge(df_cadastro, on="ID_CLIENTE", how="left")
        .merge(df_hist, on=["ID_CLIENTE", "SAFRA_REF"], how="left")
    )


# Função para remover o dominio do email
def adjust_email_domain(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["DOMINIO_EMAIL"], errors="ignore")


def fill_null_values_after_merge(df: pd.DataFrame) -> pd.DataFrame:
    cols_grp = {"PORTE", "SEGMENTO_INDUSTRIAL"}
    if cols_grp.issubset(df.columns):
        # Preenche com mediana (RENDA_MES_ANTERIOR)
        df["RENDA_MES_ANTERIOR"] = (
            df.groupby(list(cols_grp))["RENDA_MES_ANTERIOR"]
            .transform(lambda x: x.fillna(x.median()))
        )
        # Caso ainda tenha NA após isso, preencher com 0
        df["RENDA_MES_ANTERIOR"] = df["RENDA_MES_ANTERIOR"].fillna(0)

        # Preenche com média (NO_FUNCIONARIOS)
        df["NO_FUNCIONARIOS"] = (
            df.groupby(list(cols_grp))["NO_FUNCIONARIOS"]
            .transform(lambda x: x.fillna(x.mean()))
        )
        # Preencher possíveis NaNs restantes com 0
        df["NO_FUNCIONARIOS"] = df["NO_FUNCIONARIOS"].fillna(0)
        # Arredondar e converter para int
        df["NO_FUNCIONARIOS"] = df["NO_FUNCIONARIOS"].round(0).astype(int)

    return df




# Função para mapear a região do CEP
def map_regiao_cep(cep2):
    try:
        cep2 = int(cep2)
    except:
        return 'Desconhecido'

    if 1 <= cep2 <= 39:
        return 'Sudeste'
    elif cep2 in set(range(40, 50)) | set(range(60, 68)) | {67}:
        return 'Nordeste'
    elif cep2 in {68, 69, 77}:
        return 'Norte'
    elif cep2 in {70, 71, 72, 73, 74, 75, 76}:
        return 'Centro-Oeste'
    elif 80 <= cep2 <= 89:
        return 'Sul'
    else:
        return 'Desconhecido'

# Função para aplicar o mapeamento e remover coluna antiga
def map_and_drop_regiao_cep(df: pd.DataFrame) -> pd.DataFrame:
    df['REGIAO_CEP'] = df['CEP_2_DIG'].apply(map_regiao_cep)
    df = df.drop(columns=['CEP_2_DIG'])
    return df

# Função para aplicar enconding
def apply_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=['REGIAO_CEP'], drop_first=True)
    df['PORTE'] = df['PORTE'].astype('category').cat.codes
    df = pd.get_dummies(df, columns=['SEGMENTO_INDUSTRIAL'], drop_first=True)
    print(df.shape)
    return df

