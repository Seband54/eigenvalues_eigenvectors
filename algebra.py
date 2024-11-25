import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Datos simulados con más usuarios y hoteles
data = {
    'usuario': ['usuario_1', 'usuario_1', 'usuario_1', 'usuario_2', 'usuario_2', 'usuario_2',
                'usuario_3', 'usuario_3', 'usuario_3', 'usuario_4', 'usuario_4', 'usuario_4',
                'usuario_5', 'usuario_5', 'usuario_5', 'usuario_6', 'usuario_6', 'usuario_6',
                'usuario_7', 'usuario_7', 'usuario_7', 'usuario_8', 'usuario_8', 'usuario_8',
                'usuario_9', 'usuario_9', 'usuario_9', 'usuario_10', 'usuario_10', 'usuario_10',
                'usuario_11', 'usuario_11', 'usuario_11', 'usuario_12', 'usuario_12', 'usuario_12',
                'usuario_13', 'usuario_13', 'usuario_13', 'usuario_14', 'usuario_14', 'usuario_14',
                'usuario_15', 'usuario_15', 'usuario_15', 'usuario_16', 'usuario_16', 'usuario_16',
                'usuario_17', 'usuario_17', 'usuario_17', 'usuario_18', 'usuario_18', 'usuario_18',
                'usuario_19', 'usuario_19', 'usuario_19', 'usuario_20', 'usuario_20', 'usuario_20',
                'usuario_21', 'usuario_21', 'usuario_21', 'usuario_22', 'usuario_22', 'usuario_22',
                'usuario_23', 'usuario_23', 'usuario_23', 'usuario_24', 'usuario_24', 'usuario_24',
                'usuario_25', 'usuario_25', 'usuario_25', 'usuario_26', 'usuario_26', 'usuario_26',
                'usuario_27', 'usuario_27', 'usuario_27', 'usuario_28', 'usuario_28', 'usuario_28',
                'usuario_29', 'usuario_29', 'usuario_29', 'usuario_30', 'usuario_30', 'usuario_30'][:60],
    'hotel': ['hotel_A', 'hotel_B', 'hotel_C', 'hotel_A', 'hotel_B', 'hotel_C',
              'hotel_B', 'hotel_C', 'hotel_D', 'hotel_A', 'hotel_B', 'hotel_C',
              'hotel_C', 'hotel_D', 'hotel_E', 'hotel_A', 'hotel_B', 'hotel_D',
              'hotel_A', 'hotel_C', 'hotel_D', 'hotel_E', 'hotel_B', 'hotel_A',
              'hotel_A', 'hotel_B', 'hotel_C', 'hotel_B', 'hotel_C', 'hotel_D',
              'hotel_A', 'hotel_B', 'hotel_C', 'hotel_C', 'hotel_D', 'hotel_E',
              'hotel_B', 'hotel_A', 'hotel_A', 'hotel_B', 'hotel_C', 'hotel_D',
              'hotel_C', 'hotel_D', 'hotel_E', 'hotel_A', 'hotel_B', 'hotel_C',
              'hotel_A', 'hotel_B', 'hotel_C', 'hotel_B', 'hotel_C', 'hotel_D',
              'hotel_A', 'hotel_B', 'hotel_C', 'hotel_C', 'hotel_D', 'hotel_E',
              'hotel_B', 'hotel_A', 'hotel_A', 'hotel_B', 'hotel_C', 'hotel_D',
              'hotel_B', 'hotel_C', 'hotel_D', 'hotel_E', 'hotel_A', 'hotel_B'][:60],
    'rating': [5, 3, 4, 4, 2, 5, 3, 5, 1, 2, 3, 4, 4, 5, 2, 5, 3, 4, 1, 5, 3, 4, 2, 5,
               5, 4, 1, 3, 2, 4, 3, 4, 5, 2, 5, 4, 1, 5, 3, 5, 2, 4, 1, 3, 4, 2, 5, 1,
               4, 3, 2, 5, 5, 3, 4, 2, 5, 1, 4, 3, 2, 5, 5, 4, 1, 3, 2, 4, 5, 4, 1, 5,
               3, 5, 2, 4, 1, 2, 5, 3, 5, 4, 2, 5, 3, 1, 4, 3, 5, 2, 4, 5, 1, 2, 5, 3][:60]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Crear la matriz de interacción usuario-hotel
usuario_hotel = df.pivot_table(index='usuario', columns='hotel', values='rating', fill_value=0)

# Normalizar las calificaciones restando la media de cada usuario
usuario_hotel_normalizado = usuario_hotel.subtract(usuario_hotel.mean(axis=1), axis=0)

# Paso 1: Aplicar PCA para reducir la dimensionalidad de la matriz
pca = PCA(n_components=3)  # Usar 3 componentes principales
usuario_hotel_pca = pca.fit_transform(usuario_hotel_normalizado)

# Paso 2: Calcular la similitud entre usuarios (en el espacio reducido)
similitud = cosine_similarity(usuario_hotel_pca)

# Paso 3: Recomendación basada en usuarios similares ponderados
def recomendar_hoteles(usuario_id, num_recomendaciones=3):
    # Encuentra el índice del usuario
    idx = usuario_hotel.index.get_loc(usuario_id)
    
    # Obtener las similitudes de este usuario con los demás
    similitudes_usuario = similitud[idx]
    
    # Ordenar los usuarios según la similitud (mayor a menor)
    usuarios_similares = np.argsort(similitudes_usuario)[::-1]
    
    # Seleccionar hoteles no calificados por el usuario pero recomendados por usuarios similares
    hoteles_recomendados = {}
    
    for usuario in usuarios_similares:
        if usuario != idx:  # Excluir al usuario mismo
            hoteles_no_calificados = usuario_hotel.iloc[usuario].where(usuario_hotel.iloc[usuario] == 0).dropna().index
            for hotel in hoteles_no_calificados:
                # Ponderar la recomendación según la similitud
                if hotel not in hoteles_recomendados:
                    hoteles_recomendados[hotel] = similitudes_usuario[usuario]
                else:
                    hoteles_recomendados[hotel] += similitudes_usuario[usuario]
        
        if len(hoteles_recomendados) >= num_recomendaciones:
            break
    
    # Ordenar por la puntuación total y seleccionar los más recomendados
    hoteles_recomendados = sorted(hoteles_recomendados.items(), key=lambda x: x[1], reverse=True)
    
    return [hotel[0] for hotel in hoteles_recomendados[:num_recomendaciones]]

# Recomendaciones para cada usuario
for usuario in usuario_hotel.index:
    hoteles_recomendados = recomendar_hoteles(usuario, num_recomendaciones=3)
    print(f"Hoteles recomendados para '{usuario}': {hoteles_recomendados}")

# Mostrar los autovectores y autovalores de PCA
print("\nAutovectores de PCA (componentes principales):")
print(pca.components_)

print("\nAutovalores de PCA (varianza explicada):")
print(pca.explained_variance_)

# Paso 4: Calcular los 3 mejores hoteles según las recomendaciones dinámicas
# Obtener las recomendaciones para todos los usuarios
todas_las_recomendaciones = {}

for usuario in usuario_hotel.index:
    recomendaciones_usuario = recomendar_hoteles(usuario, num_recomendaciones=3)
    todas_las_recomendaciones[usuario] = recomendaciones_usuario

# Contar cuántas veces se recomienda cada hotel
todos_los_hoteles_recomendados = [hotel for hoteles in todas_las_recomendaciones.values() for hotel in hoteles]
conteo_hoteles = Counter(todos_los_hoteles_recomendados)

# Ordenar los hoteles por número de recomendaciones (de mayor a menor)
hoteles_mejor_recomendados = conteo_hoteles.most_common()

# Mostrar los 3 mejores hoteles recomendados
print("\nLos 3 mejores hoteles recomendados basados en las preferencias de los usuarios:")
for i, (hotel, conteo) in enumerate(hoteles_mejor_recomendados[:3], 1):
    print(f"{i}. {hotel} (Recomendado {conteo} veces)")
