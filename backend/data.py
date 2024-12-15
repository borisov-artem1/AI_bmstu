from itertools import product
from math import isnan

import pandas as pd
import itertools
import numpy as np
from IPython.core.display_functions import display
from numpy.f2py.auxfuncs import throw_error
from numpy.linalg import norm
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy.sparse import random

pd.set_option('display.max_columns', None)
product_df = pd.read_csv("./data/protein_dataset.csv")

flavors_df = pd.read_csv("./data/flavors_tree_with_characteristics.csv")
df = product_df.copy(deep=True)

flavors_df["category"] = flavors_df["category"].map(lambda value: str(value).lower())
flavors_df["flavor"] = flavors_df["flavor"].map(lambda value: str(value).lower() + ', ') + \
    flavors_df["characteristics"].map(lambda value: ', '.join(map(str, str(value).lower().split(', '))))

flavors_df = flavors_df.groupby("category", as_index=False).agg({
    "flavor": ', '.join,
    "characteristics": ', '.join,
})

flavors_df["flavor"] = flavors_df["flavor"].apply(lambda value: set(value.split(', ')))
flavors_df["characteristics"] = flavors_df["characteristics"].apply(lambda value: set(value.split(', ')))

exploded_df = flavors_df.explode("flavor").explode("characteristics")

flavor_dummies = pd.get_dummies(exploded_df["flavor"]).groupby(exploded_df["category"]).max()
characteristics_dummies = pd.get_dummies(exploded_df["characteristics"]).groupby(exploded_df["category"]).max()

flavor_df = pd.concat([flavor_dummies, characteristics_dummies], axis=1).fillna(0).astype(int).reset_index()

# Записал для наглядности в csv файл
flavor_df_file = 'data/flavor_df.csv'
with open(flavor_df_file, mode='w', newline='', encoding='utf-8') as file:
    flavor_df.to_csv(file, index=False)


# Кодирование категорийных данных
# -----------------------------------------------------------
type_dict = {
    "Protein" : 0, "Gainer" : 0.2,
    "Mass protein" : 0.4, "Casein" : 0.6,
    "Whey" : 0.8, "Isolate" : 1
}
df["type"] = df["type"].map(lambda value: type_dict[value])

# -----------------------------------------------------------
is_vegan_dict = { False: 0, True: 1}
df["is_vegan"] = df["is_vegan"].map(lambda value: is_vegan_dict[value])

# -----------------------------------------------------------
country_dict = {
    "Italy" : 0, "France" : 1,
    "USA" : 2, "Russia" : 3,
    "United Kingdom" : 4, "Spanish" : 5,
    "Poland" : 6, "Hawaii" : 7,
    "Germany" : 8
}
# -----------------------------------------------------------
intensity_dict = {
    "extremely low" : 0, "low" : 0.25,
    "medium" : 0.5, "high" : 0.75,
    "extremely high" : 1
}
df["intensity"] = df["intensity"].map(lambda value: intensity_dict[value])

# -----------------------------------------------------------

families = ["fruity", "nutty", "dessert", "nan"]

first_tree_layer_dict = { value: index for index, value in enumerate(families) }

first_tree_layer_similarity = np.zeros((len(first_tree_layer_dict), len(first_tree_layer_dict)))

first_tree_layer_similarity[first_tree_layer_dict["Fruity"]][first_tree_layer_dict["Nutty"]] = first_tree_layer_similarity[first_tree_layer_dict["Nutty"]][first_tree_layer_dict["Fruity"]] = 0.6
first_tree_layer_similarity[first_tree_layer_dict["Fruity"]][first_tree_layer_dict["Dessert"]] = first_tree_layer_similarity[first_tree_layer_dict["Dessert"]][first_tree_layer_dict["Fruity"]] = 0.9
first_tree_layer_similarity[first_tree_layer_dict["Fruity"]][first_tree_layer_dict["nan"]] = first_tree_layer_similarity[first_tree_layer_dict["nan"]][first_tree_layer_dict["Fruity"]] = 0.1

first_tree_layer_similarity[first_tree_layer_dict["Nutty"]][first_tree_layer_dict["Dessert"]] = first_tree_layer_similarity[first_tree_layer_dict["Dessert"]][first_tree_layer_dict["Nutty"]] = 1
first_tree_layer_similarity[first_tree_layer_dict["Nutty"]][first_tree_layer_dict["nan"]] = first_tree_layer_similarity[first_tree_layer_dict["nan"]][first_tree_layer_dict["Nutty"]] = 0.1

first_tree_layer_similarity[first_tree_layer_dict["Dessert"]][first_tree_layer_dict["nan"]] = first_tree_layer_similarity[first_tree_layer_dict["nan"]][first_tree_layer_dict["Dessert"]] = 0.1


subfamilies = [
    "tropical fruits", "citrus fruits", "stone fruits",
    "nan", "chocolate-based"
]


second_tree_layer_dict = { value: index for index, value in enumerate(subfamilies) }
second_tree_layer_similarity = np.zeros((len(second_tree_layer_dict), len(second_tree_layer_dict)))


second_tree_layer_similarity[second_tree_layer_dict["Tropical Fruits"]][second_tree_layer_dict["Citrus Fruits"]] = 0.7
second_tree_layer_similarity[second_tree_layer_dict["Tropical Fruits"]][second_tree_layer_dict["Stone Fruits"]] = 0.6
second_tree_layer_similarity[second_tree_layer_dict["Tropical Fruits"]][second_tree_layer_dict["Chocolate-based"]] = 0.3
second_tree_layer_similarity[second_tree_layer_dict["Tropical Fruits"]][second_tree_layer_dict["nan"]] = 0.2

second_tree_layer_similarity[second_tree_layer_dict["Citrus Fruits"]][second_tree_layer_dict["Stone Fruits"]] = 0.5
second_tree_layer_similarity[second_tree_layer_dict["Citrus Fruits"]][second_tree_layer_dict["nan"]] = 0.2
second_tree_layer_similarity[second_tree_layer_dict["Citrus Fruits"]][second_tree_layer_dict["Chocolate-based"]] = 0.3

second_tree_layer_similarity[second_tree_layer_dict["Stone Fruits"]][second_tree_layer_dict["Chocolate-based"]] = 0.3
second_tree_layer_similarity[second_tree_layer_dict["Stone Fruits"]][second_tree_layer_dict["nan"]] = 0.2

second_tree_layer_similarity[second_tree_layer_dict["Chocolate-based"]][second_tree_layer_dict["nan"]] = 0.1

# Завершаем симметричность матрицы
for i in range(len(subfamilies)):
    for j in range(i + 1, len(subfamilies)):
        second_tree_layer_similarity[j][i] = second_tree_layer_similarity[i][j]

for i in range(len(subfamilies)):
    for j in range(len(subfamilies)):
        if i == j:
            second_tree_layer_similarity[i][j] = 1



flavors = [
    "chocolate", "vanilla", "cookies & cream", "banana",
    "mango", "peanut butter", "coconut", "caramel",
    "hazelnut", "pineapple", "lemon", "orange",
    "apple", "peach", "pistachio", "watermelon",
    "tropical punch", "nan"
]

third_tree_layer_dict = {value: index for index, value in enumerate(flavors)}
third_tree_layer_similarity = np.zeros((len(third_tree_layer_dict), len(third_tree_layer_dict)))

# Заполняем матрицу значениями для всех возможных пар
# Заполняем матрицу сходства для всех вкусов
for i, flavor1 in enumerate(flavors):
    for j, flavor2 in enumerate(flavors):
        if i == j:
            third_tree_layer_similarity[i][j] = 1.0  # Максимальная схожесть с самим собой
        elif "None" in [flavor1, flavor2]:
            third_tree_layer_similarity[i][j] = 0.1  # Минимальное сходство с "None"
        else:
            # Пример распределения значений на основе категорий
            if flavor1 in ["Chocolate", "Coconut", "Cookies & Cream", "Vanilla", "Caramel"] and \
               flavor2 in ["Chocolate", "Coconut", "Cookies & Cream", "Vanilla", "Caramel"]:
                third_tree_layer_similarity[i][j] = 0.8  # Десертные
            elif flavor1 in ["Banana", "Mango", "Pineapple", "Tropical Punch", "Watermelon"] and \
                 flavor2 in ["Banana", "Mango", "Pineapple", "Tropical Punch", "Watermelon"]:
                third_tree_layer_similarity[i][j] = 0.8  # Тропические фрукты
            elif flavor1 in ["Apple", "Peach"] and flavor2 in ["Apple", "Peach"]:
                third_tree_layer_similarity[i][j] = 0.8  # Косточковые
            elif flavor1 in ["Lemon", "Orange"] and flavor2 in ["Lemon", "Orange"]:
                third_tree_layer_similarity[i][j] = 0.8  # Цитрусовые
            elif flavor1 in ["Peanut Butter", "Hazelnut", "Pistachio"] and \
                 flavor2 in ["Peanut Butter", "Hazelnut", "Pistachio"]:
                third_tree_layer_similarity[i][j] = 0.8  # Ореховые
            elif flavor1 in ["Chocolate", "Cookies & Cream", "Vanilla"] and \
                 flavor2 in ["Peanut Butter", "Hazelnut", "Pistachio"]:
                third_tree_layer_similarity[i][j] = 0.6  # Десертные и Ореховые
            elif flavor1 in ["Banana", "Mango", "Pineapple", "Tropical Punch", "Watermelon"] and \
                 flavor2 in ["Apple", "Peach"]:
                third_tree_layer_similarity[i][j] = 0.5  # Тропические и Косточковые
            elif flavor1 in ["Banana", "Mango", "Pineapple", "Tropical Punch", "Watermelon"] and \
                 flavor2 in ["Lemon", "Orange"]:
                third_tree_layer_similarity[i][j] = 0.4  # Тропические и Цитрусовые
            else:
                third_tree_layer_similarity[i][j] = 0.3

layer = [first_tree_layer_dict, second_tree_layer_dict, third_tree_layer_dict]
tree = [first_tree_layer_similarity, second_tree_layer_similarity, third_tree_layer_similarity]



country_matr = np.zeros((len(country_dict), len(country_dict)))

# Заполнение матрицы
for country1, i in country_dict.items():
    for country2, j in country_dict.items():
        if i == j:
            country_matr[i][j] = 1.0  # Сходство с собой
        else:
            # Установим значения на основе логики сходства
            if country1 in ["Italy", "France", "Germany", "Spain", "Poland"] and \
               country2 in ["Italy", "France", "Germany", "Spain", "Poland"]:
                country_matr[i][j] = 0.8  # Европейские страны

            elif (country1, country2) in [("USA", "United Kingdom"), ("United Kingdom", "USA")]:
                country_matr[i][j] = 0.7  # Связанные исторически и экономически

            elif (country1, country2) in [("Russia", "Poland"), ("Poland", "Russia")]:
                country_matr[i][j] = 0.6  # Близкие культурно и географически

            elif country1 in ["USA", "Hawaii"] and country2 in ["USA", "Hawaii"]:
                country_matr[i][j] = 0.5  # США и Гавайи

            elif country1 in ["Italy", "France", "Germany", "Spain", "Poland"] and country2 == "Russia":
                country_matr[i][j] = 0.4  # Европа и Россия

            elif country1 in ["Hawaii"] or country2 in ["Hawaii"]:
                country_matr[i][j] = 0.3  # Гавайи и другие

            else:
                country_matr[i][j] = 0.2  # Остальные пары имеют низкое сходство


df["unit_price"] = df["price_rub"] / df["weight_g"]
df["unit_price"] = (df["unit_price"].values - min(df["unit_price"].values)) / (max(df["unit_price"].values) - min(df["unit_price"].values))
del df["price_rub"]
del df["weight_g"]

flavor_tree = [
    # Tropical Fruits
    {"family": "Fruity", "category": "Tropical Fruits", "flavor": "Banana", "characteristics": "sweet, soft, creamy"},
    {"family": "Fruity", "category": "Tropical Fruits", "flavor": "Mango", "characteristics": "sweet, juicy, exotic"},
    {"family": "Fruity", "category": "Tropical Fruits", "flavor": "Pineapple", "characteristics": "tangy, refreshing, tropical"},
    {"family": "Fruity", "category": "Tropical Fruits", "flavor": "Tropical Punch", "characteristics": "sweet, fruity, exotic"},

    # Citrus Fruits
    {"family": "Fruity", "category": "Citrus Fruits", "flavor": "Lemon", "characteristics": "sour, refreshing, citrusy"},
    {"family": "Fruity", "category": "Citrus Fruits", "flavor": "Orange", "characteristics": "sweet, citrusy, uplifting"},

    # Stone Fruits
    {"family": "Fruity", "category": "Stone Fruits", "flavor": "Peach", "characteristics": "sweet, juicy, soft"},
    {"family": "Fruity", "category": "Stone Fruits", "flavor": "Watermelon", "characteristics": "sweet, refreshing, summery"},
    {"family": "Fruity", "category": "Stone Fruits", "flavor": "Apple", "characteristics": "sweet, crisp, refreshing"},

    # Nut-based
    {"family": "Nutty", "category": "nan", "flavor": "Peanut Butter", "characteristics": "nutty, creamy, rich"},
    {"family": "Nutty", "category": "nan", "flavor": "Hazelnut", "characteristics": "nutty, sweet, rich"},
    {"family": "Nutty", "category": "nan", "flavor": "Pistachio", "characteristics": "nutty, slightly sweet, mild"},

    # Chocolate-based
    {"family": "Dessert", "category": "Chocolate-based", "flavor": "Chocolate", "characteristics": "bitter-sweet, rich, creamy"},
    {"family": "Dessert", "category": "Chocolate-based", "flavor": "Coconut", "characteristics": "creamy, sweet, nutty"},
    {"family": "Dessert", "category": "Chocolate-based", "flavor": "Cookies & Cream", "characteristics": "sweet, creamy, chocolatey"},

    # Vanilla-based
    {"family": "Dessert", "category": "nan", "flavor": "Vanilla", "characteristics": "sweet, creamy, floral"},

    # Caramel-based
    {"family": "Dessert", "category": "nan", "flavor": "Caramel", "characteristics": "sweet, caramelized, creamy"},

    # Miscellaneous
    {"family": "nan", "category": "nan", "flavor": "nan", "characteristics": "neutral, undefined"}
]

# Создаем словарь с flavor как ключом
flavor_dict = {item["flavor"]: item for item in flavor_tree}

# Применяем map для добавления столбцов
df["family"] = df["flavor"].map(lambda x: flavor_dict.get(x, {}).get("family", "nan"))
df["category"] = df["flavor"].map(lambda x: flavor_dict.get(x, {}).get("category", "nan"))


df["brand"] = df["brand"].map(lambda value: str(value).lower())
df["country"] = df["country"].map(lambda value: str(value).lower())
df["flavor"] = df["flavor"].map(lambda value: str(value).lower())
df["family"] = df["family"].map(lambda value: str(value).lower())
df["category"] = df["category"].map(lambda value: str(value).lower())

display(df)
merged_df = df.merge(flavor_df, on="category", how="left", suffixes=('', '_category'))
display(merged_df)

df_tree = pd.DataFrame({
    'brand': df['brand'],
    'flavor_info': df[['family', 'category', 'flavor']].values.tolist()
})


exclude_fields = [
    "brand", "country", "type", "is_vegan",
    "unit_price", "category", "flavor", "family"
]

def get_df_flavors(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy(deep=True)
    for elem in exclude_fields:
        del new_df[elem]
    return new_df

def get_stat(df: pd.DataFrame) -> pd.DataFrame:
    new_df = pd.DataFrame(columns=exclude_fields, data=df[exclude_fields].values)
    del new_df["country"]
    del new_df["brand"]
    del new_df["flavor"]
    del new_df["family"]
    del new_df["category"]
    return new_df

def _complete(first_vec, second_vec, elem):
    for elem in range(len(first_vec) - len(second_vec)):
        second_vec.append(elem)
    return second_vec


def complete(first_vec, second_vec, elem):
    if len(first_vec) > len(second_vec):
        second_vec = _complete(first_vec, second_vec, elem)
    elif len(first_vec) < len(second_vec):
        first_vec = _complete(second_vec, first_vec, elem)
    return first_vec, second_vec

def get_dist(first_vec, second_vec, power):
    result = 0
    for i in range(len(first_vec)):
        if isnan(first_vec[i]) or isnan(second_vec[i]):
            continue
        result += pow(first_vec[i] + second_vec[i], power)
    power /= power ** 2
    result = pow(result, power)
    return result

def manhattan_dist(first_vec, second_vec):
    return get_dist(first_vec, second_vec, 1)

def euclid_dist(first_vec, second_vec):
    return get_dist(first_vec, second_vec, 2)

def get_cos_dist(first_vec: list, second_vec: list):
    result = 0
    complete(first_vec, second_vec, 0)
    divisible = [first_vec[i] * second_vec[i] for i in range(len(first_vec))]
    first_divider = [pow(first_vec[i], 2) for i in range(len(first_vec))]
    second_divider = [pow(second_vec[i], 2) for i in range(len(second_vec))]
    first_divider = pow(sum(first_divider), 0.5)
    second_divider = pow(sum(second_divider), 0.5)
    divisible = sum(divisible)
    result = divisible / (first_divider * second_divider)
    return result

def tree_dist(first_vec, second_vec):
    result = 0
    if len(first_vec) != len(second_vec):
        complete(first_vec, second_vec, 'nan')

    for i in range(len(first_vec)):
        result += tree[i][layer[i][first_vec[i]]][layer[i][second_vec[i]]]

    return result / len(tree)

def get_brand_dist(first_vec, second_vec):
    return 1 if first_vec[0] == second_vec[0] else 0

def get_country_dist(first_val, second_val):
    return country_matr[country_dict[first_val]][country_dict[second_val]]

# Мера Жаккара
def _get_jac(v1, v2):
    a = v1.count(1)
    b = v2.count(1)
    c = 0
    for i in range(len(v1)):
        if v1[i] and v2[i]:
            c += 1
    return 1 - c / (a + b - c)


def get_jac(data_f):
    matr_data = data_f.values.tolist()
    n = len(matr_data)
    res_matr = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            res_matr[i][j] = res_matr[j][i] = _get_jac(matr_data[i], matr_data[j])
    return res_matr


def calc_distance(f, data_f):
    data_matr = data_f.values.tolist()
    n = len(data_matr)
    res_matr = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            res_matr[i][j] = res_matr[j][i] = f(data_matr[i], res_matr[j])
    return res_matr / res_matr.max()


def calcDistanceCompined(df, dfTree):
    dfTree = dfTree["семейства"]
    dfMan = getDataFrameAroma(df)
    dfJac = dfMan.copy()
    del dfJac["шлейф"]

    dfStatParams = getDataFrameStat(df)

    matrTree = calcDistance(getTreeDistance, dfTree)
    # matrCos = calcDistance(getCos, dfMan)
    matrEucl = calcDistance(getEuclideanDistance, dfStatParams)
    # matrMan = calcDistance(getManhattanDistance, dfMan)
    matrBrand = calcDistance(getBrandDistance, df["бренд"])
    matrCountry = calcDistance(getCountryDistance, df["страна"])
    matrJac = getJacquard(dfJac, nameArr)

    xTree = matrTree.max()
    # xCos = matrCos.max()
    xEuci = matrEucl.max()
    # xMan = matrMan.max()
    xJac = matrJac.max()
    xBrand = matrBrand.max()
    xCountry = matrCountry.max()

    kJac, kTree, kEuci, kBrand, kCountry = 2, 0.5, 10, 2, 2

    return (kJac * matrJac + kTree * matrTree + matrEucl + kBrand * matrBrand + kCountry * matrCountry) / (
                kJac * xJac + kTree * xTree + xEuci + kBrand * xBrand + kCountry * xCountry)





