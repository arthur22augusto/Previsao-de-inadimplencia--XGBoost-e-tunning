import pandas as pd
import numpy as np

# Função usada para criar novas colunas na base de pagamentos -- Importante
def create_new_columns_pgto(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.assign(
        DIAS_ATE_PAGAMENTO=lambda d: (
            d["DATA_PAGAMENTO"] - d["DATA_EMISSAO_DOCUMENTO"]
        ).dt.days,
        INADIMPLENTE=lambda d: np.where(
            (d["DATA_PAGAMENTO"] - d["DATA_VENCIMENTO"]).dt.days >= 5, 1, 0
        ),
        DIAS_DE_ATRASO=lambda d: (
            d["DATA_PAGAMENTO"] - d["DATA_VENCIMENTO"]
        ).dt.days,
        PRAZO_PAGAMENTO_DIAS=lambda d: (
            d["DATA_VENCIMENTO"] - d["DATA_EMISSAO_DOCUMENTO"]
        ).dt.days,
    )
    return df


# Versão para inferência (sem DATA_PAGAMENTO)
def create_new_columns_pgto_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DIAS_ATE_PAGAMENTO"] = np.nan
    df["INADIMPLENTE"] = np.nan
    df["DIAS_DE_ATRASO"] = np.nan
    df["PRAZO_PAGAMENTO_DIAS"] = (
        df["DATA_VENCIMENTO"] - df["DATA_EMISSAO_DOCUMENTO"]
    ).dt.days
    return df

#Função usada para criar features historicas -- IMPORTANTE
def create_hist_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.sort_values(by=["ID_CLIENTE", "SAFRA_REF"]).copy()
    grp = df_feat.groupby("ID_CLIENTE")

    df_feat["PAGOU_ADIANTADO"] = (df_feat["DIAS_DE_ATRASO"] < 0).astype(int)
    df_feat["HIST_PAGAMENTOS_ADIANTADOS"] = grp["PAGOU_ADIANTADO"].cumsum().shift(1)
    df_feat["HIST_DIAS_ATRASO_ACUMULADO"] = grp["DIAS_DE_ATRASO"].cumsum().shift(1)

    soma_atraso_3m = (
        grp["DIAS_DE_ATRASO"].rolling(window=3, min_periods=1).sum()
        .reset_index(level=0, drop=True)
    )
    df_feat["HIST_DIAS_ATRASO_3M"] = soma_atraso_3m.shift(1)

    df_feat["HIST_QTD_TRANSACOES"] = grp.cumcount()
    df_feat["HIST_VALOR_MEDIO_PAGO"] = (
        grp["VALOR_A_PAGAR"].expanding().mean()
        .reset_index(level=0, drop=True)
        .shift(1)
    )
    df_feat["HIST_INADIMPLENTE_ULT_MES"] = grp["INADIMPLENTE"].shift(1)
    df_feat["HIST_MEDIA_DIAS_ATRASO"] = (
        grp["DIAS_DE_ATRASO"].expanding().mean()
        .reset_index(level=0, drop=True)
        .shift(1)
    )

    cols_hist = [c for c in df_feat.columns if c.startswith("HIST_")]
    df_feat[cols_hist] = df_feat[cols_hist].fillna(0)

    return df_feat