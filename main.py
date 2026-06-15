from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import os

app = FastAPI(title="Basketball Analytics - Modelo de Predicción")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cargar modelo ──────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_prediccion.pkl")
datos = joblib.load(MODEL_PATH)

modelo_final     = datos['modelo_final']
usar_scaler      = datos['usar_scaler']
scaler           = datos['scaler']
feature_cols     = datos['feature_cols']
nba_feature_cols = datos['nba_feature_cols']
all_targets      = datos['all_targets']
le_nba           = datos['le_nba']
le_acb           = datos['le_acb']
medias_dataset   = datos['medias_dataset']

_X_dataset      = datos['X_dataset']
_scaler_sim     = MinMaxScaler().fit(_X_dataset)
_X_dataset_norm = _scaler_sim.transform(_X_dataset)
_nombres_temp   = datos['nombres_temporadas'].reset_index(drop=True)

COLS_PORCENTAJE = [
    'Porcentaje de acierto tiros de 3 puntos',
    'Porcentaje de acierto tiros de 2 puntos',
    'Porcentaje de acierto tiros libres',
]
POSICIONES_NBA_VALIDAS = ['G', 'F', 'C']
POSICIONES_ACB_VALIDAS = ['Base', 'Escolta', 'Alero', 'Ala-pívot', 'Pívot']

# ── Input schema ───────────────────────────────────────────────────────
class PlayerInput(BaseModel):
    posicion_nba: str = 'F'
    altura_metros: float = 2.00
    posicion_acb: str = 'Alero'
    points: Optional[float] = None
    assists: Optional[float] = None
    fieldGoalsMade: Optional[float] = None
    fieldGoalsAttempted: Optional[float] = None
    fieldGoalsPercentage: Optional[float] = None
    threePointersMade: Optional[float] = None
    threePointersAttempted: Optional[float] = None
    threePointersPercentage: Optional[float] = None
    freeThrowsMade: Optional[float] = None
    freeThrowsAttempted: Optional[float] = None
    freeThrowsPercentage: Optional[float] = None
    reboundsTotal: Optional[float] = None
    reboundsOffensive: Optional[float] = None
    reboundsDefensive: Optional[float] = None
    steals: Optional[float] = None
    blocks: Optional[float] = None
    turnovers: Optional[float] = None
    foulsPersonal: Optional[float] = None
    plusMinusPoints: Optional[float] = None
    minutos: Optional[float] = None
    trueShootingPercentage: Optional[float] = None
    effectiveFieldGoalPercentage: Optional[float] = None
    usagePercentage: Optional[float] = None
    assistPercentage: Optional[float] = None
    offensiveReboundPercentage: Optional[float] = None
    defensiveReboundPercentage: Optional[float] = None
    playerImpactEstimate: Optional[float] = None
    assistToTurnoverRatio: Optional[float] = None
    offensiveRating: Optional[float] = None
    defensiveRating: Optional[float] = None
    netRating: Optional[float] = None

# ── Endpoint de predicción ─────────────────────────────────────────────
@app.post("/predecir")
def predecir(data: PlayerInput):
    stats = data.dict()
    posicion_nba  = stats.pop('posicion_nba')
    altura_metros = stats.pop('altura_metros')
    posicion_acb  = stats.pop('posicion_acb')

    # Validar posiciones
    if posicion_nba not in POSICIONES_NBA_VALIDAS:
        posicion_nba = 'F'
    if posicion_acb not in POSICIONES_ACB_VALIDAS:
        posicion_acb = 'Alero'

    # Construir vector de features
    fila = {
        col: (stats.get(col.replace('nba_', '')) or medias_dataset[col])
        for col in nba_feature_cols
    }
    fila['posicion_nba_enc'] = le_nba.transform([posicion_nba])[0]
    fila['altura_m_fija']    = float(altura_metros)
    fila['posicion_acb_enc'] = le_acb.transform([posicion_acb])[0]

    X_nuevo = np.array([[fila[c] for c in feature_cols]])
    if usar_scaler:
        X_nuevo = scaler.transform(X_nuevo)

    prediccion = modelo_final.predict(X_nuevo)[0]

    targets_mostrar = [t for t in all_targets if t.endswith('_pg') or t in COLS_PORCENTAJE]
    idx_mostrar     = [all_targets.index(t) for t in targets_mostrar]
    nombres_limpios = [t.replace('_pg', '') for t in targets_mostrar]

    predicciones = [
        {"estadistica": nombres_limpios[i], "valor": round(float(prediccion[idx_mostrar[i]]), 2)}
        for i in range(len(nombres_limpios))
    ]

    # Jugadores similares
    X_nuevo_norm = _scaler_sim.transform(X_nuevo)
    distancias   = cdist(X_nuevo_norm, _X_dataset_norm, metric='euclidean')[0]
    df_sim = _nombres_temp.copy()
    df_sim['distancia'] = distancias.round(3)
    df_sim = (
        df_sim.sort_values('distancia')
        .drop_duplicates(subset='nombre_completo')
        .head(5)
        .reset_index(drop=True)
    )
    similares = [
        {"jugador": row['nombre_completo'], "temporada": str(row['Temporada']), "distancia": row['distancia']}
        for _, row in df_sim.iterrows()
    ]

    return {"predicciones": predicciones, "similares": similares}

@app.get("/")
def root():
    return {"status": "Basketball Analytics API funcionando"}
