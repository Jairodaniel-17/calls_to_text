# @title Comparación de cadenas de texto "similaridad[0-1]"
import nltk


def comparar_cadenas(cadena_1: str, cadena_2: str) -> float:
    """
    Esta función compara dos cadenas de texto y
    devuelve un índice de similitud basado en el índice de Jaccard.

    Parámetros:
    cadena_1 (str): La primera cadena de texto a comparar.
    cadena_2 (str): La segunda cadena de texto a comparar.

    Devuelve:
    ratio (float): Un número entre 0 y 1 que representa el índice de similitud entre las dos cadenas de texto.
                   Un valor de 1 significa que las cadenas son idénticas, y un valor de 0 significa que no tienen nada en común.

    Ejemplo:
    >>> comparar_cadenas("Hola, ¿cómo estás?", "Hola, ¿qué tal?")
    0.20833333333333334
    """
    set1 = set(nltk.ngrams(cadena_1, n=3))
    set2 = set(nltk.ngrams(cadena_2, n=3))
    ratio = len(set1.intersection(set2)) / len(set1.union(set2))
    return ratio
