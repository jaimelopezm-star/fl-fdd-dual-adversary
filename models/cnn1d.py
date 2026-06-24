"""
cnn1d.py — Modelo base 1D-CNN (estilo WDCNN) para diagnóstico de fallas sobre señal de vibración.

WDCNN (Zhang et al., 2017) es el clasificador 1D-CNN de referencia para CWRU: una primera
convolución de kernel ANCHO (64) que actúa como filtro de banda sobre la señal cruda, seguida de
convoluciones estrechas. Aquí la entrada tiene 2 canales (Drive-End y Fan-End), porque el
Mecanismo 1 (Opción C) hará consenso entre ambos acelerómetros.

    Entrada:  (batch, 2, 2048)   -> 2 canales DE/FE, ventana de 2048 muestras
    Salida:   (batch, n_classes) -> logits (0=Normal,1=InnerRace,2=OuterRace,3=Ball)
"""
from __future__ import annotations
import torch.nn as nn


class WDCNN(nn.Module):
    """1D-CNN tipo WDCNN. Primer kernel ancho (64, stride 16) + bloques conv estrechos."""

    def __init__(self, in_channels: int = 2, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            # bloque 1: kernel ANCHO sobre señal cruda
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            # bloque 2
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            # bloque 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            # bloque 4
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            # bloque 5
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 100), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(100, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(in_channels: int = 2, n_classes: int = 4) -> WDCNN:
    """Fábrica de modelos. Útil para pasar `model_fn` al loop FL (cada cliente reinstancia)."""
    return WDCNN(in_channels=in_channels, n_classes=n_classes)
