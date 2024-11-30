"""
Módulo para analizar los resultados del preprocesamiento.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from .data_structures import TennisPoint

def analyze_results(points: List[TennisPoint]) -> Dict[str, Any]:
    """
    Analiza los puntos detectados y genera estadísticas.
    
    Args:
        points: Lista de puntos a analizar
    
    Returns:
        Diccionario con estadísticas y métricas
    """
    if not points:
        return {
            "error": "No hay puntos para analizar",
            "stats": None
        }
    
    # Calcular estadísticas básicas
    durations = [p.duration for p in points]
    confidences = [p.confidence for p in points]
    
    stats = {
        'total_points': len(points),
        'duration': {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'min': min(durations),
            'max': max(durations),
            'median': np.median(durations)
        },
        'confidence': {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'median': np.median(confidences)
        },
        'points_with_score': sum(1 for p in points if p.score_shown),
        'score_percentage': (sum(1 for p in points if p.score_shown) / len(points)) * 100
    }
    
    return {
        "stats": stats,
        "visualization": create_visualizations(points)
    }

def create_visualizations(points: List[TennisPoint]) -> Dict[str, plt.Figure]:
    """
    Crea visualizaciones de los puntos analizados.
    
    Args:
        points: Lista de puntos a visualizar
    
    Returns:
        Diccionario con las figuras generadas
    """
    durations = [p.duration for p in points]
    confidences = [p.confidence for p in points]
    
    # Configurar estilo
    plt.style.use('seaborn')
    
    # Figura 1: Distribuciones
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Distribución de duraciones
    sns.histplot(durations, bins=20, ax=ax1)
    ax1.set_title('Distribución de Duraciones')
    ax1.set_xlabel('Duración (segundos)')
    ax1.set_ylabel('Frecuencia')
    
    # Distribución de confianzas
    sns.histplot(confidences, bins=20, ax=ax2)
    ax2.set_title('Distribución de Confianzas')
    ax2.set_xlabel('Confianza')
    ax2.set_ylabel('Frecuencia')
    
    # Figura 2: Análisis temporal
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Timeline de puntos
    for i, point in enumerate(points):
        ax3.plot([point.start_time, point.end_time], [i, i], '-')
    ax3.set_title('Timeline de Puntos')
    ax3.set_xlabel('Tiempo (segundos)')
    ax3.set_ylabel('Número de Punto')
    
    # Duración vs Confianza
    ax4.scatter(durations, confidences, alpha=0.6)
    ax4.set_xlabel('Duración (segundos)')
    ax4.set_ylabel('Confianza')
    ax4.set_title('Duración vs Confianza')
    
    plt.tight_layout()
    
    return {
        "distributions": fig1,
        "temporal_analysis": fig2
    }

def print_analysis_summary(stats: Dict[str, Any]):
    """
    Imprime un resumen del análisis en formato legible.
    
    Args:
        stats: Diccionario con estadísticas
    """
    print("\nResumen del Análisis")
    print("=" * 50)
    print(f"Total de puntos detectados: {stats['total_points']}")
    print(f"\nDuración de puntos (segundos):")
    print(f"  Media: {stats['duration']['mean']:.2f} ± {stats['duration']['std']:.2f}")
    print(f"  Mediana: {stats['duration']['median']:.2f}")
    print(f"  Rango: [{stats['duration']['min']:.2f}, {stats['duration']['max']:.2f}]")
    
    print(f"\nConfianza de detección:")
    print(f"  Media: {stats['confidence']['mean']:.2f} ± {stats['confidence']['std']:.2f}")
    print(f"  Mediana: {stats['confidence']['median']:.2f}")
    print(f"  Rango: [{stats['confidence']['min']:.2f}, {stats['confidence']['max']:.2f}]")
    
    print(f"\nPuntos con marcador visible: {stats['points_with_score']}")
    print(f"Porcentaje con marcador: {stats['score_percentage']:.1f}%")