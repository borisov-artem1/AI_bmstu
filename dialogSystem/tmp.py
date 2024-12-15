from itertools import product
from math import isnan
import plotly.express as px
import pandas as pd
import itertools
import numpy as np
from cffi.pkgconfig import merge_flags
from numpy import dot
from IPython.core.display_functions import display
from numpy.f2py.auxfuncs import throw_error
from numpy.linalg import norm
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy.sparse import random

# from data.filling_the_dataset import intensity
# %%
pd.set_option('display.max_columns', None)
product_df = pd.read_csv("./data/protein_dataset.csv")

flavors_df = pd.read_csv("./data/flavors_tree_with_characteristics.csv")
df = product_df.copy(deep=True)
display(df, flavors_df)
# %%
flavors_df["category"] = flavors_df["category"].map(lambda value: str(value).lower())
flavors_df["flavor"] = flavors_df["flavor"].map(lambda value: str(value).lower() + ', ') + \
                       flavors_df["characteristics"].map(
                           lambda value: ', '.join(map(str, str(value).lower().split(', '))))

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


type_dict = {
    "Protein": 0, "Gainer": 0.2,
    "Mass protein": 0.4, "Casein": 0.6,
    "Whey": 0.8, "Isolate": 1
}
df["type"] = df["type"].map(lambda value: type_dict[value])

# -----------------------------------------------------------
is_vegan_dict = {False: 0, True: 1}
df["is_vegan"] = df["is_vegan"].map(lambda value: is_vegan_dict[value])

# -----------------------------------------------------------
country_dict = {
    "italy": 0, "france": 1,
    "usa": 2, "russia": 3,
    "united kingdom": 4, "spanish": 5,
    "poland": 6, "hawaii": 7,
    "germany": 8
}
# -----------------------------------------------------------
intensity_dict = {
    "extremely low": 0, "low": 0.25,
    "medium": 0.5, "high": 0.75,
    "extremely high": 1
}
df["intensity"] = df["intensity"].map(lambda value: intensity_dict[value])

# -----------------------------------------------------------

families = ["fruity", "nutty", "dessert", "nan"]

first_tree_layer_dict = {value: index for index, value in enumerate(families)}
print(first_tree_layer_dict)
first_tree_layer_similarity = np.zeros((len(first_tree_layer_dict), len(first_tree_layer_dict)))

first_tree_layer_similarity[first_tree_layer_dict["fruity"]][first_tree_layer_dict["nutty"]] = \
first_tree_layer_similarity[first_tree_layer_dict["nutty"]][first_tree_layer_dict["fruity"]] = 0.6
first_tree_layer_similarity[first_tree_layer_dict["fruity"]][first_tree_layer_dict["dessert"]] = \
first_tree_layer_similarity[first_tree_layer_dict["dessert"]][first_tree_layer_dict["fruity"]] = 0.9
first_tree_layer_similarity[first_tree_layer_dict["fruity"]][first_tree_layer_dict["nan"]] = \
first_tree_layer_similarity[first_tree_layer_dict["nan"]][first_tree_layer_dict["fruity"]] = 0.1

first_tree_layer_similarity[first_tree_layer_dict["nutty"]][first_tree_layer_dict["dessert"]] = \
first_tree_layer_similarity[first_tree_layer_dict["dessert"]][first_tree_layer_dict["nutty"]] = 1
first_tree_layer_similarity[first_tree_layer_dict["nutty"]][first_tree_layer_dict["nan"]] = \
first_tree_layer_similarity[first_tree_layer_dict["nan"]][first_tree_layer_dict["nutty"]] = 0.1

first_tree_layer_similarity[first_tree_layer_dict["dessert"]][first_tree_layer_dict["nan"]] = \
first_tree_layer_similarity[first_tree_layer_dict["nan"]][first_tree_layer_dict["dessert"]] = 0.1

# %%
subfamilies = [
    "tropical fruits", "citrus fruits", "stone fruits",
    "nan", "chocolate-based"
]

second_tree_layer_dict = {value: index for index, value in enumerate(subfamilies)}
second_tree_layer_similarity = np.zeros((len(second_tree_layer_dict), len(second_tree_layer_dict)))

second_tree_layer_similarity[second_tree_layer_dict["tropical fruits"]][second_tree_layer_dict["citrus fruits"]] = 0.7
second_tree_layer_similarity[second_tree_layer_dict["tropical fruits"]][second_tree_layer_dict["stone fruits"]] = 0.6
second_tree_layer_similarity[second_tree_layer_dict["tropical fruits"]][second_tree_layer_dict["chocolate-based"]] = 0.3
second_tree_layer_similarity[second_tree_layer_dict["tropical fruits"]][second_tree_layer_dict["nan"]] = 0.2

second_tree_layer_similarity[second_tree_layer_dict["citrus fruits"]][second_tree_layer_dict["stone fruits"]] = 0.5
second_tree_layer_similarity[second_tree_layer_dict["citrus fruits"]][second_tree_layer_dict["nan"]] = 0.2
second_tree_layer_similarity[second_tree_layer_dict["citrus fruits"]][second_tree_layer_dict["chocolate-based"]] = 0.3

second_tree_layer_similarity[second_tree_layer_dict["stone fruits"]][second_tree_layer_dict["chocolate-based"]] = 0.3
second_tree_layer_similarity[second_tree_layer_dict["stone fruits"]][second_tree_layer_dict["nan"]] = 0.2

second_tree_layer_similarity[second_tree_layer_dict["chocolate-based"]][second_tree_layer_dict["nan"]] = 0.1

# Завершаем симметричность матрицы
for i in range(len(subfamilies)):
    for j in range(i + 1, len(subfamilies)):
        second_tree_layer_similarity[j][i] = second_tree_layer_similarity[i][j]

for i in range(len(subfamilies)):
    for j in range(len(subfamilies)):
        if i == j:
            second_tree_layer_similarity[i][j] = 1.0
# %%
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
for i, flavor1 in enumerate(flavors):
    for j, flavor2 in enumerate(flavors):
        if i == j:
            third_tree_layer_similarity[i][j] = 1.0  # Максимальная схожесть с самим собой
        elif "nan" in [flavor1, flavor2]:
            third_tree_layer_similarity[i][j] = 0.1  # Минимальное сходство с "None"
        else:
            # Пример распределения значений на основе категорий
            if flavor1 in ["chocolate", "coconut", "cookies & cream", "vanilla", "caramel"] and \
                    flavor2 in ["chocolate", "coconut", "cookies & cream", "vanilla", "caramel"]:
                third_tree_layer_similarity[i][j] = 0.8  # Десертные
            elif flavor1 in ["banana", "mango", "pineapple", "tropical punch", "watermelon"] and \
                    flavor2 in ["banana", "mango", "pineapple", "tropical punch", "watermelon"]:
                third_tree_layer_similarity[i][j] = 0.8  # Тропические фрукты
            elif flavor1 in ["apple", "peach"] and flavor2 in ["apple", "peach"]:
                third_tree_layer_similarity[i][j] = 0.8  # Косточковые
            elif flavor1 in ["lemon", "orange"] and flavor2 in ["lemon", "orange"]:
                third_tree_layer_similarity[i][j] = 0.8  # Цитрусовые
            elif flavor1 in ["peanut butter", "hazelnut", "pistachio"] and \
                    flavor2 in ["peanut butter", "hazelnut", "pistachio"]:
                third_tree_layer_similarity[i][j] = 0.8  # Ореховые
            elif flavor1 in ["chocolate", "cookies & cream", "vanilla"] and \
                    flavor2 in ["peanut butter", "hazelnut", "pistachio"]:
                third_tree_layer_similarity[i][j] = 0.6  # Десертные и Ореховые
            elif flavor1 in ["banana", "mango", "pineapple", "tropical punch", "watermelon"] and \
                    flavor2 in ["apple", "peach"]:
                third_tree_layer_similarity[i][j] = 0.5  # Тропические и Косточковые
            elif flavor1 in ["banana", "mango", "pineapple", "tropical punch", "watermelon"] and \
                    flavor2 in ["lemon", "orange"]:
                third_tree_layer_similarity[i][j] = 0.4  # Тропические и Цитрусовые
            else:
                third_tree_layer_similarity[i][j] = 0.3

layer = [first_tree_layer_dict, second_tree_layer_dict, third_tree_layer_dict]
tree = [first_tree_layer_similarity, second_tree_layer_similarity, third_tree_layer_similarity]
# %%
country_matr = np.zeros((len(country_dict), len(country_dict)))

# Заполнение матрицы
for country1, i in country_dict.items():
    for country2, j in country_dict.items():
        if i == j:
            country_matr[i][j] = 1.0  # Сходство с собой
        else:
            # Установим значения на основе логики сходства
            if country1 in ["italy", "france", "germany", "spain", "poland"] and \
                    country2 in ["italy", "france", "germany", "spain", "poland"]:
                country_matr[i][j] = 0.8  # Европейские страны

            elif (country1, country2) in [("usa", "united kingdom"), ("united kingdom", "usa")]:
                country_matr[i][j] = 0.7  # Связанные исторически и экономически

            elif (country1, country2) in [("russia", "poland"), ("poland", "russia")]:
                country_matr[i][j] = 0.6  # Близкие культурно и географически

            elif country1 in ["usa", "hawaii"] and country2 in ["usa", "hawaii"]:
                country_matr[i][j] = 0.5  # США и Гавайи

            elif country1 in ["italy", "france", "germany", "spain", "poland"] and country2 == "russia":
                country_matr[i][j] = 0.4  # Европа и Россия

            elif country1 in ["hawaii"] or country2 in ["hawaii"]:
                country_matr[i][j] = 0.3  # Гавайи и другие

            else:
                country_matr[i][j] = 0.2  # Остальные пары имеют низкое сходство
# %%
df["unit_price"] = df["price_rub"] / df["weight_g"]
df["unit_price"] = (df["unit_price"].values - min(df["unit_price"].values)) / (
            max(df["unit_price"].values) - min(df["unit_price"].values))
del df["price_rub"]
del df["weight_g"]

# %%
flavor_tree = [
    # Tropical Fruits
    {"family": "Fruity", "category": "Tropical Fruits", "flavor": "Banana", "characteristics": "sweet, soft, creamy"},
    {"family": "Fruity", "category": "Tropical Fruits", "flavor": "Mango", "characteristics": "sweet, juicy, exotic"},
    {"family": "Fruity", "category": "Tropical Fruits", "flavor": "Pineapple",
     "characteristics": "tangy, refreshing, tropical"},
    {"family": "Fruity", "category": "Tropical Fruits", "flavor": "Tropical Punch",
     "characteristics": "sweet, fruity, exotic"},

    # Citrus Fruits
    {"family": "Fruity", "category": "Citrus Fruits", "flavor": "Lemon",
     "characteristics": "sour, refreshing, citrusy"},
    {"family": "Fruity", "category": "Citrus Fruits", "flavor": "Orange",
     "characteristics": "sweet, citrusy, uplifting"},

    # Stone Fruits
    {"family": "Fruity", "category": "Stone Fruits", "flavor": "Peach", "characteristics": "sweet, juicy, soft"},
    {"family": "Fruity", "category": "Stone Fruits", "flavor": "Watermelon",
     "characteristics": "sweet, refreshing, summery"},
    {"family": "Fruity", "category": "Stone Fruits", "flavor": "Apple", "characteristics": "sweet, crisp, refreshing"},

    # Nut-based
    {"family": "Nutty", "category": "nan", "flavor": "Peanut Butter", "characteristics": "nutty, creamy, rich"},
    {"family": "Nutty", "category": "nan", "flavor": "Hazelnut", "characteristics": "nutty, sweet, rich"},
    {"family": "Nutty", "category": "nan", "flavor": "Pistachio", "characteristics": "nutty, slightly sweet, mild"},

    # Chocolate-based
    {"family": "Dessert", "category": "Chocolate-based", "flavor": "Chocolate",
     "characteristics": "bitter-sweet, rich, creamy"},
    {"family": "Dessert", "category": "Chocolate-based", "flavor": "Coconut",
     "characteristics": "creamy, sweet, nutty"},
    {"family": "Dessert", "category": "Chocolate-based", "flavor": "Cookies & Cream",
     "characteristics": "sweet, creamy, chocolatey"},

    # Vanilla-based
    {"family": "Dessert", "category": "nan", "flavor": "Vanilla", "characteristics": "sweet, creamy, floral"},

    # Caramel-based
    {"family": "Dessert", "category": "nan", "flavor": "Caramel", "characteristics": "sweet, caramelized, creamy"},

    # Miscellaneous
    {"family": "nan", "category": "nan", "flavor": "nan", "characteristics": "neutral, undefined"}
]

flavor_to_family = {item["flavor"].lower(): item["family"].lower() for item in flavor_tree}
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
merged_df = df.merge(flavor_df, on="category", how="left", suffixes=('', '_category'))

name_arr = df["brand"]
df_tree = pd.DataFrame({
    'brand': df['brand'],
    'flavor_info': merged_df[['family', 'category', 'flavor']].values.tolist()
})
values_for_flavor_df_dessert = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0]


# %%
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


get_df_flavors(merged_df)

# %%
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
    res = 0
    for i in range(len(first_vec)):
        if isnan(first_vec[i]) or isnan(second_vec[i]):
            continue
        res += pow(abs(first_vec[i] - second_vec[i]), power)
    return pow(res, 1 / power)


# %%
def manhattan_dist(first_vec, second_vec):
    return get_dist(first_vec, second_vec, 1)


def euclid_dist(first_vec, second_vec):
    return get_dist(first_vec, second_vec, 2)


def get_cos_dist(v1, v2):
    # Создаем копии векторов и фильтруем NaN элементы
    v1_filtered = [val for val, other in zip(v1, v2) if not isnan(val) and not isnan(other)]
    v2_filtered = [val for val, other in zip(v2, v1) if not isnan(val) and not isnan(other)]

    # Проверка, если после фильтрации один из векторов пустой
    if len(v1_filtered) == 0 or len(v2_filtered) == 0:
        return 1  # Вернуть максимальное расстояние, если нечего сравнивать

    # Вычисляем норму для проверки нулевых векторов
    norm_v1 = norm(v1_filtered)
    norm_v2 = norm(v2_filtered)

    # Если хотя бы один из векторов нулевой, возвращаем максимальное расстояние
    if norm_v1 == 0 or norm_v2 == 0:
        return 1

    # Вычисляем косинусное расстояние
    cos_similarity = dot(v1_filtered, v2_filtered) / (norm_v1 * norm_v2)
    return 1 - cos_similarity  # Конвертируем косинусное сходство в расстояние


def tree_dist(first_vec, second_vec):
    result = 0
    if len(first_vec) != len(second_vec):
        complete(first_vec, second_vec, 'nan')

    for i in range(len(first_vec)):
        result += tree[i][layer[i][first_vec[i]]][layer[i][second_vec[i]]]

    return 1 - result / len(tree)


# %%
def get_brand_dist(first_vec, second_vec):
    return 1 if first_vec[0] == second_vec[0] else 0


def get_country_dist(first_val, second_val):
    return 1 - country_matr[country_dict[first_val]][country_dict[second_val]]


# Мера Жаккара
def _get_jac(v1, v2):
    # Преобразуем входные данные в множества для корректного выполнения операции
    set1, set2 = set(v1), set(v2)

    # Вычисляем пересечение и объединение множеств
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Проверка, чтобы избежать деления на 0
    if union == 0:
        return 0  # Вернем 0, если оба множества пусты

    # Вычисляем коэффициент Жаккара
    jaccard_index = intersection / union
    return jaccard_index


def get_jac(data_f):
    matr_data = data_f.values.tolist()
    n = len(matr_data)
    res_matr = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            res_matr[i][j] = res_matr[j][i] = _get_jac(matr_data[i], matr_data[j])
    return res_matr


# %%
def calc_distance(f, data_f):
    data_matr = data_f.values.tolist()
    n = len(data_matr)
    res_matr = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            res_matr[i][j] = res_matr[j][i] = f(data_matr[i], data_matr[j])
    return res_matr / res_matr.max()


def calcDistanceCompined(df, dfTree):
    dfTree = dfTree["flavor_info"]
    dfMan = get_df_flavors(df)

    dfStatParams = get_stat(df)

    matrTree = calc_distance(tree_dist, dfTree)
    matrEucl = calc_distance(euclid_dist, dfStatParams)
    matrMan = calc_distance(manhattan_dist, dfMan)
    matrBrand = calc_distance(get_brand_dist, df["brand"])
    matrCountry = calc_distance(get_country_dist, df["country"])

    xTree = matrTree.max()
    xEuci = matrEucl.max()
    xMan = matrMan.max()
    xBrand = matrBrand.max()
    xCountry = matrCountry.max()

    kMan, kTree, kEuci, kBrand, kCountry = 10, 5, 10, 2, 2

    return (kMan * matrMan + kTree * matrTree + matrEucl + kBrand * matrBrand + kCountry * matrCountry) / (
            kMan * xMan + kTree * xTree + xEuci + kBrand * xBrand + kCountry * xCountry)


# %%
def draw(matrRes, nameArr, title, color='Inferno'):
    fig = px.imshow(matrRes, x=nameArr, y=nameArr, color_continuous_scale=color, title=title)
    fig.update_layout(width=1000, height=1200)
    fig.update_traces(text=nameArr)
    fig.update_xaxes(side="top")
    fig.show()


# %%
res_matr = calc_distance(manhattan_dist, get_df_flavors(merged_df))
#draw(res_matr, name_arr, "Manhattan distance")

# %%
res_matr = calc_distance(euclid_dist, get_stat(merged_df))
#draw(res_matr, name_arr, "Euclidian distance")
# %%
res_matr = calc_distance(tree_dist, df_tree["flavor_info"])
#draw(res_matr, name_arr, "Tree distance")
# %%
res_matr = calc_distance(get_country_dist, df["country"])
#draw(res_matr, name_arr, "Country distance")


#draw(calcDistanceCompined(df, df_tree), name_arr, "Combined distance")


F_NAME = "brand"
F_DIST = "distance"
matrSimilarity = calcDistanceCompined(df, df_tree)


def printRes(arr):
    print("distance \t\t\t brand")
    for elem in arr:
        for key, value in elem.items():
            print("{0}\t\t{1}".format(value, key))


def getSimilarity(id, matr, name_arr):
    data = matr[id]
    res = pd.DataFrame(zip(data, name_arr), index=np.arange(len(matr)), columns=["distance", "brand"])
    return res.sort_values("distance")


def _findSimilar(name):
    print(name)

    if not name.istitle():
        name = name.capitalize()
        if name == "Bsn" or name == "Gls" or name == "Animal":
            name = name.upper()
        if name == "Evlution nutrition":
            name = "EVLution Nutrition"
        if name == "Ultimate nutrition":
            name = "Ultimate Nutrition"
        if name == "Muscletech":
            name = "MuscleTech"
        if name == "Musclemeds":
            name = "MuscleMeds"
        if name == "Big snt":
            name = "Big SNT"
        if name == "Metabolic nutrition":
            name = "Metabolic Nutrition"
        if name == "Garden of life":
            name = "Garden of Life"
        if name == "Asp nutrition":
            name = "ASP Nutrition"
        if name == "Insane labz":
            name = "Insane Labz"
        if name == "Optimum nutrition":
            name = "Optimum Nutrition"
        if name == "Bucked up":
            name = "Bucked UP"
        if name == "Olimp sport nutrition":
            name = "Olimp Sport Nutrition"
        if name == "Usn nutrition":
            name = "USN Nutrition"
        if name == "Geneticlab nutrition":
            name = "Geneticlab Nutrition"
        if name == "California gold nutrition":
            name = "California Gold Nutrition"

    ind = product_df[F_NAME].tolist().index(name)

    listSimilarity = getSimilarity(ind, matrSimilarity, name_arr)
    return listSimilarity


def findSimilar(name):
    listSimilarity = _findSimilar(name)

    return listSimilarity[listSimilarity[F_NAME] != name.lower()]


res = findSimilar("Allmax")



from collections import defaultdict


def _findSimilarMany(name_arr):
    recList = []
    for name in name_arr:
        rec = _findSimilar(name)
        recList.append(rec.loc[rec[F_NAME].isin(name_arr) == False])

    dfRes = defaultdict(lambda: 1e2)
    for rec in recList:
        for i, row in rec.iterrows():
            curName = row[F_NAME]
            curDist = row[F_DIST]
            dfRes[curName] = min(dfRes[curName], curDist)

    return dfRes


def findSimilarMany(name_arr):
    resDict = _findSimilarMany(name_arr)
    name_arr = [elem.lower() for elem in name_arr]
    return sorted([{key: elem} for key, elem in resDict.items() if key not in name_arr],
                  key=lambda elem: list(elem.values())[0])


res = findSimilarMany(["Allmax", "BSN"])

def delOpposite(dict, name_arr):
    for name in name_arr:
        if name in dict.keys():
            del dict[name]

    return dict


def findSimilar(likesArr, dislikesArr):
    likesRec = delOpposite(_findSimilarMany(likesArr), dislikesArr)
    dislikesRec = delOpposite(_findSimilarMany(dislikesArr), likesArr)

    dictRes = {}
    if len(likesArr) == 0:
        for key, elem in dislikesRec.items():
            if elem > 0.7:
                dictRes[key] = elem
        return sorted([{key: elem} for key, elem in dictRes.items()], key=lambda elem: list(elem.values())[0])

    for key in likesRec.keys():
        if likesRec[key] <= dislikesRec[key]:
            dictRes[key] = likesRec[key]
    return sorted([{key: elem} for key, elem in dictRes.items()], key=lambda elem: list(elem.values())[0])


res = findSimilar(["BSN"],
                  ["Optimum Nutrition"])
printRes(res)

merged_df['dessert'] = values_for_flavor_df_dessert
import ipywidgets as widgets
from tkinter import font
from turtle import onclick, width

import panel as pn

pn.extension('tabulator')


def getArrFromSeries(data):
    arr = []
    for elem in data:
        arr.append(elem)
    return arr


def getDataFrameFromArr(data, reverse=0):
    resArr = []
    for elem in data:
        for key in elem.keys():
            resArr.append({"Название": key, "Величина схожести": 1 - elem[key] + reverse * (2 * elem[key] - 1)})

    return pd.DataFrame(resArr, index=range(1, len(resArr) + 1), columns=["Название", "Величина схожести"])


namesUI = getArrFromSeries(name_arr)

choiceLiked = pn.widgets.MultiChoice(
    name='👍👍👍 Нравится 👍👍👍',
    value=[],
    width=320,
    options=namesUI)

choiceDisliked = pn.widgets.MultiChoice(
    name='👎👎👎 НЕ нравится 👎👎👎',
    value=[],
    width=320,
    options=namesUI)


markdownError = pn.pane.Markdown(
    '### <div style="font-family: serif; text-align: center; color: red;">Ошибка ввода</div>',
    width=800, visible=False
)
markdownDefault = pn.pane.Markdown("#### Выберете то, что: ", width=800, visible=True)
markdownResultMustTitle = pn.pane.Markdown("#### Попробуйте следующие протеины: ", width=300, visible=False)
markdownResultMaybeTitle = pn.pane.Markdown("#### Возможно Вам понравятся: ", width=300, visible=False)

bokeh_formatters = {
    "Величина схожести": {'type': 'progress', 'max': 1}
}

tableRecMust = pn.widgets.Tabulator(visible=False, formatters=bokeh_formatters)
tableRecMaybe = pn.widgets.Tabulator(visible=False, formatters=bokeh_formatters)



def _initMustTable(recArr):
    tableRecMust.value = getDataFrameFromArr(recArr)

    markdownResultMustTitle.visible = True
    tableRecMust.visible = True


def _initMaybeTable(recArr):
    tableRecMaybe.value = getDataFrameFromArr(recArr)

    markdownResultMaybeTitle.visible = True
    tableRecMaybe.visible = True


def _changeStatusError(isError):
    if isError:
        markdownError.visible = True
        markdownDefault.visible = False
        markdownResultMustTitle.visible = False
        markdownResultMaybeTitle.visible = False

        tableRecMust.visible = False
        tableRecMaybe.visible = False
    else:
        markdownError.visible = False
        markdownDefault.visible = True


def _splitMustMaybe(recArr):
    recMust, recMaybe = [], []
    for rec in recArr:
        for key in rec.keys():
            if rec[key] <= 0.5:
                recMust.append(rec)
            else:
                recMaybe.append(rec)
    return recMust, recMaybe


def _splitMustMaybeDict(recDict):
    recMust, recMaybe = [], []
    for name, value in recDict.items():
        if value >= 0.5:
            recMust.append({name: value})
        else:
            recMaybe.append({name: value})
    return recMust, recMaybe


def _isRightInput(arr1, arr2):
    inner = list(set(arr1) & set(arr2))
    return len(inner) == 0


def _getDefaultResult(nameArr):
    resArr = []
    for name in nameArr:
        resArr.append({name: 1})
    return resArr


def _getDefaultResultParams(nameArr):
    resArr = {}
    for name in nameArr:
        resArr[name] = 0
    return resArr


def _getRecommendationArr(likesArr, dislikesArr):
    recArr = None

    if len(likesArr) and len(dislikesArr):
        recArr = findSimilar(likesArr, dislikesArr)
    elif len(likesArr) and len(dislikesArr) == 0:
        recArr = findSimilarMany(likesArr)
    elif len(likesArr) == 0 and len(dislikesArr):
        recArr = findSimilar(likesArr, dislikesArr)
    else:
        recArr = _getDefaultResult(namesUI)
    return recArr


def _giveRecommendation(likesArr, dislikesArr):
    recArr = _getRecommendationArr(likesArr, dislikesArr)

    recMust, recMaybe = _splitMustMaybe(recArr)

    _initMustTable(recMust)
    _initMaybeTable(recMaybe)


def run(a):
    likesArr = choiceLiked.value
    dislikesArr = choiceDisliked.value

    if not _isRightInput(likesArr, dislikesArr):
        _changeStatusError(isError=True)
        return

    _changeStatusError(isError=False)
    _giveRecommendation(likesArr, dislikesArr)


button = pn.widgets.Button(
    name='Готово',
    button_type='success',
    width=50,
    height=40,
    margin=(24, 100, 10, 10))
button.on_click(run)

pLikes = pn.Column(
    markdownDefault,
    markdownError,
    pn.Row(
        pn.Column(choiceLiked, height=800),
        pn.Column(choiceDisliked, height=800),
        button,
        pn.Column(markdownResultMustTitle,
                  tableRecMust,
                  markdownResultMaybeTitle,
                  tableRecMaybe))
)


from IPython.core.display import HTML

HTML("""
<style>
.bk.custom-card {
    background-color: WhiteSmoke;
    border-radius: 5px;
    padding: 10px;
}
</style>
""")

intensityWidget = pn.widgets.CheckBoxGroup(
    name='Интенсивность',
    options=["extremely low", "low", "medium", "high", "extremely high"],
    inline=False
)
intensityElem = pn.Card(
    intensityWidget,
    title='Тип',
    css_classes=['custom-card'],
    width=400,
    margin=(10, 30, 10, 10))

countryArr = list(set(product_df['country'].tolist()))

countryWidget = pn.widgets.MultiChoice(
    value=[],
    options=countryArr)
countryElem = pn.Card(
    countryWidget,
    title='Страна-производитель',
    css_classes=['custom-card'],
    width=400,
    margin=(10, 30, 10, 10))

brandArr = list(set(product_df['brand'].tolist()))

brandWidget = pn.widgets.MultiChoice(
    value=[],
    options=brandArr)
brandElem = pn.Card(
    brandWidget,
    title='Бренд',
    css_classes=['custom-card'],
    width=400,
    margin=(10, 30, 10, 10))

typeArr = list(set(product_df['type'].tolist()))

typeWidget = pn.widgets.MultiChoice(
    value=[],
    options=typeArr)
typeElem = pn.Card(
    typeWidget,
    title='Тип продукта',
    css_classes=['custom-card'],
    width=400,
    margin=(10, 30, 10, 10))

families = ["fruity", "nutty", "dessert", "none"]
#familyArrT = product_df['flavor'].map(lambda elem: elem if isinstance(elem, str) else 'None').tolist()

#familyArrT = list(set(familyArrT))
familyArrT = families

familyWidget = pn.widgets.MultiChoice(
    value=[],
    options=[elem.lower() for elem in familyArrT])
familyElem = pn.Card(
    familyWidget,
    title='Вкусы',
    css_classes=['custom-card'],
    width=400,
    margin=(10, 30, 10, 10))


flavor_df_list = flavor_df.columns.values.tolist()
display(flavor_df_list)
flavors = [
    "Chocolate", "Vanilla", "Cookies & Cream", "Banana",
    "Mango", "Peanut Butter", "Coconut", "Caramel",
    "Hazelnut", "Pineapple", "Lemon", "Orange",
    "Apple", "Peach", "Pistachio", "Watermelon",
    "Tropical Punch", "nan"
]
flavors = [elem.lower() for elem in flavors]
#display(flavors)
flavor_df_list.remove('category')
#display(flavor_df)
#display(flavor_df_list)
#tasteArr = list(set(flavor_df_list) | set(flavors))
tasteArr = list(set(flavors))
tasteArr.sort()

tasteWidget = pn.widgets.MultiSelect(
    value=[],
    size=10,
    options=tasteArr)
tasteElem = pn.Card(
    tasteWidget,
    title='Вкусы',
    css_classes=['custom-card'],
    width=400,
    margin=(10, 60, 10, 10))

markdownResultMustTitle2 = pn.pane.Markdown("#### Попробуйте следующие вкусы: ", width=300, visible=False)
markdownResultMaybeTitle2 = pn.pane.Markdown("#### Возможно вам понравятся: ", width=300, visible=False)

tableRecMust2 = pn.widgets.Tabulator(visible=False, formatters=bokeh_formatters)
tableRecMaybe2 = pn.widgets.Tabulator(visible=False, formatters=bokeh_formatters)


def _initMustTable2(recArr):
    tableRecMust2.value = getDataFrameFromArr(recArr, 1)

    markdownResultMustTitle2.visible = True
    tableRecMust2.visible = True


def _initMaybeTable2(recArr):
    tableRecMaybe2.value = getDataFrameFromArr(recArr, 1)

    markdownResultMaybeTitle2.visible = True
    tableRecMaybe2.visible = True


def _updateIntense(intenseArr):
    res = []
    for intensity in intenseArr:
        res.append(intensity_dict[intensity])
    return res


def _updateType(typeArr):
    res = []
    for type in typeArr:
        res.append(type_dict[type])
    return res


def sortDict(resDict):
    sorted_tuples = sorted(resDict.items(), key=lambda item: item[1], reverse=True)
    sorted_dict = {k: v for k, v in sorted_tuples}
    return sorted_dict


def _updateResult(dataDict, nameArr, n):
    resDict = {}
    for key, value in dataDict.items():
        if value == 0:
            continue
        resDict[nameArr[key]] = value / n

    return sortDict(resDict)


def _fromArrDictToDict(arrDict):
    resDict = {}
    for elem in arrDict:
        resDict.update(elem)
    return resDict


def _getRecommendationParams(intensitySelected, countrySelected, brandSelected,
                             typeSelected, familySelected, tasteSelected, merged_df):
    dfColumnsArr = merged_df.columns.values.tolist()
    indexDict = {}
    indexDict[dfColumnsArr.index('intensity')] = intensitySelected
    indexDict[dfColumnsArr.index('country')] = countrySelected
    indexDict[dfColumnsArr.index('brand')] = brandSelected
    indexDict[dfColumnsArr.index('type')] = typeSelected
    print(dfColumnsArr)
    for family in familySelected:
        if family == 'none':
            family = 'nan'
        indexDict[dfColumnsArr.index(family)] = [1]
    #print(indexDict)
    for taste in tasteSelected:
        if taste == 'none':
            taste = 'nan'
        indexDict[dfColumnsArr.index(taste)] = [1]
    #print(indexDict)

    matrData = merged_df.values.tolist()
    sDict = {}
    for i in range(len(matrData)):
        s = 0
        for ind in indexDict.keys():
            if matrData[i][ind] in indexDict[ind]:
                s += 1
        sDict[i] = s
    #print(sDict)
    return sDict


def getRecommendationParams(intensitySelected, countrySelected, brandSelected,
                            typeSelected, familySelected, tasteSelected, merged_df):
    nAll = 0
    if len(intensitySelected):
        nAll += 1
    if len(countrySelected):
        nAll += 1
    if len(brandSelected):
        nAll += 1
    if len(typeSelected):
        nAll += 1

    nAll += len(familySelected) + len(tasteSelected)
    if nAll == 0:
        recDict = _getDefaultResultParams(namesUI)
    else:
        recDict = _getRecommendationParams(intensitySelected, countrySelected, brandSelected, typeSelected,
                                           familySelected, tasteSelected, merged_df)
        recDict = _updateResult(recDict, name_arr, nAll)
    return recDict






def _compareLikesParams(likesDict, paramsDict, familySelected, tasteSelected, dislikesSelected):
    resDict = {}
    likesDictRes = {}

    if len(familySelected) != 0:
        for family in familySelected:
            for like in likesDict.keys():
                brand_data = merged_df[merged_df['brand'] == like]
                if family in brand_data['family'].values:
                    likesDictRes[like] = likesDict[like]

    if len(tasteSelected) != 0:
        for taste in tasteSelected:
            for like in likesDict.keys():
                brand_data = merged_df[merged_df['brand'] == like]
                if taste in brand_data['flavor'].values:
                    likesDictRes[like] = likesDict[like]
                elif flavor_to_family[taste] in brand_data['family'].values:
                    likesDictRes[like] = likesDict[like]


    for key in dislikesSelected:
        if key in paramsDict.keys():
            del paramsDict[key]

    display(paramsDict)
    display(likesDict)
    for key, value in likesDictRes.items():
        if key in paramsDict.keys():
            resDict[key] = (value + paramsDict[key]) * 0.5
            del paramsDict[key]
        else:
            resDict[key] = value * 0.5

    if len(paramsDict.keys()):
        for key, value in paramsDict.items():
            resDict[key] = 0.5 * value
    print(resDict)
    return resDict


def giveRecommendationFull(intensitySelected, countrySelected, brandSelected,
                           typeSelected, familySelected, tasteSelected, likesSelected, dislikesSelected, merged_df):
    recDict = getRecommendationParams(intensitySelected, countrySelected, brandSelected, typeSelected, familySelected,
                                      tasteSelected, merged_df)
    if len(likesSelected) or len(dislikesSelected):
        recLikesArr = _getRecommendationArr(likesSelected, dislikesSelected)
        recLikesDict = _fromArrDictToDict(recLikesArr)
        recDict = _compareLikesParams(recLikesDict, recDict, familySelected, tasteSelected, dislikesSelected)

    recMust, recMaybe = _splitMustMaybeDict(sortDict(recDict))

    _initMustTable2(recMust)
    _initMaybeTable2(recMaybe)


def runFull(a):
    intensitySelected = intensityWidget.value
    countrySelected = countryWidget.value
    brandSelected = brandWidget.value
    typeSelected = typeWidget.value
    familySelected = familyWidget.value
    tasteSelected = tasteWidget.value

    likesSelected = choiceLiked.value
    dislikesSelected = choiceDisliked.value

    if not _isRightInput(likesSelected, dislikesSelected):
        _changeStatusError(isError=True)
        return

    _changeStatusError(isError=False)

    intensitySelected = _updateIntense(intensitySelected)
    typeSelected = _updateType(typeSelected)

    giveRecommendationFull(intensitySelected, countrySelected, brandSelected, typeSelected, familySelected,
                           tasteSelected, likesSelected, dislikesSelected, merged_df)


buttonFull = pn.widgets.Button(
    name='Готово',
    button_type='success',
    width=400,
    height=40,
    margin=(24, 100, 10, 10))
buttonFull.on_click(runFull)

elemArr = pn.Column(
    intensityElem,
    countryElem,
    brandElem,
    typeElem,
    familyElem,
    tasteElem,
    buttonFull
)

elemSet = pn.Row(
    elemArr,
    pn.Column(
        markdownResultMustTitle2,
        tableRecMust2,
        markdownResultMaybeTitle2,
        tableRecMaybe2
    )
)

tabs = pn.Tabs(("👍/👎", pLikes), ("⚙️", elemSet))
tabs.show()