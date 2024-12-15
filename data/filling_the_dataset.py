import csv
import random
import os
from itertools import count

# Список вкусов из колеса вкусов для протеина/гейнера

brands = [
    "BSN", "Big SNT", "Maxler", "USN Nutrition", "Gaspari", "Optimum Nutrition",
    "Ultimate Nutrition", "Bucked UP", "Bombbar", "Geneticlab Nutrition",
    "Primekraft", "EVLution Nutrition", "Garden of Life", "California Gold Nutrition",
    "MuscleTech", "Allmax", "Metabolic Nutrition", "MuscleMeds", "Sunwarrior", "ASP Nutrition",
    "Insane Labz", "GLS", "ANIMAL", "Olimp Sport Nutrition", "REX Nutrition"
]

families = ["Fruity", "Nutty", "Dessert", "None"]

subfamilies = [
    "Tropical Fruits", "Citrus Fruits", "Stone Fruits",
    "None", "Chocolate-based"
]

flavors = [
    "Chocolate", "Vanilla", "Cookies & Cream", "Banana",
    "Mango", "Peanut Butter", "Coconut", "Caramel",
    "Hazelnut", "Pineapple", "Lemon", "Orange",
    "Apple", "Peach", "Pistachio", "Watermelon",
    "Tropical Punch", "nan"
]

country = [
    "Italy", "France", "USA", "Russia", "United Kingdom",
    "Spanish", "Poland", "Hawaii", "Germany"
]

intensity = [
    "extremely low", "low", "medium",
    "high", "extremely high"
]

proteins = [
    "Protein", "Gainer", "Mass protein",
    "Casein", "Whey", "Isolate"
]

# Данные для категорий и вкусов, включая характеристики для каждого вкуса
flavors_data = [
    # Tropical Fruits
    {"category": "Tropical Fruits", "flavor": "Banana", "characteristics": "sweet, soft, creamy"},
    {"category": "Tropical Fruits", "flavor": "Mango", "characteristics": "sweet, juicy, exotic"},
    {"category": "Tropical Fruits", "flavor": "Pineapple", "characteristics": "tangy, refreshing, tropical"},
    {"category": "Tropical Fruits", "flavor": "Tropical Punch", "characteristics": "sweet, fruity, exotic"},

    # Citrus Fruits
    {"category": "Citrus Fruits", "flavor": "Lemon", "characteristics": "sour, refreshing, citrusy"},
    {"category": "Citrus Fruits", "flavor": "Orange", "characteristics": "sweet, citrusy, uplifting"},

    # Stone Fruits
    {"category": "Stone Fruits", "flavor": "Peach", "characteristics": "sweet, juicy, soft"},
    {"category": "Stone Fruits", "flavor": "Watermelon", "characteristics": "sweet, refreshing, summery"},
    {"category": "Stone Fruits", "flavor": "Apple", "characteristics": "sweet, crisp, refreshing"},

    # Nut-based
    {"category": "Nutty", "flavor": "Peanut Butter", "characteristics": "nutty, creamy, rich"},
    {"category": "Nutty", "flavor": "Hazelnut", "characteristics": "nutty, sweet, rich"},
    {"category": "Nutty", "flavor": "Pistachio", "characteristics": "nutty, slightly sweet, mild"},

    # Chocolate-based
    {"category": "Chocolate-based", "flavor": "Chocolate", "characteristics": "bitter-sweet, rich, creamy"},
    {"category": "Chocolate-based", "flavor": "Coconut", "characteristics": "creamy, sweet, nutty"},
    {"category": "Chocolate-based", "flavor": "Cookies & Cream", "characteristics": "sweet, creamy, chocolatey"},

    # Vanilla-based
    {"category": "Dessert", "flavor": "Vanilla", "characteristics": "sweet, creamy, floral"},

    # Caramel-based
    {"category": "Dessert", "flavor": "Caramel", "characteristics": "sweet, caramelized, creamy"},

    # Miscellaneous
    {"category": "None", "flavor": "None", "characteristics": "neutral, undefined"}
]


# Путь к CSV файлу
csv_file_path = 'data/flavors_tree_with_characteristics.csv'

# Запись в CSV файл
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["category", "flavor", "characteristics"])
    writer.writeheader()  # Записываем заголовки
    writer.writerows(flavors_data)

# Генерация датасета
dataset = []
for i in range(len(brands) - 1):
    k = random.randint(0, len(brands) - 1)
    item = {
        "brand": brands[k],
        "country": random.choice(country),
        "type": random.choice(proteins),
        "is_vegan": random.choice([True, False]),
        "price_rub": random.randint(1000, 5000),
        "weight_g": random.randint(500, 2000),
        "flavor": random.choice(flavors),
        "intensity": random.choice(intensity)
        #"characteristics": flavors_data[random.randint(0, len(flavors_data) - 1)]["characteristics"]
    }
    del brands[k]
    dataset.append(item)

# Сохранение датасета в json файл
csv_file_path = 'data/protein_dataset.csv'
os.makedirs('data', exist_ok=True)

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ["brand", "country", "type", "is_vegan", "price_rub", "weight_g", "flavor", "intensity"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(dataset)

#dataset_path
