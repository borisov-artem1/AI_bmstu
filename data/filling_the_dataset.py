import json
import random

# Список вкусов из колеса вкусов для протеина/гейнера
flavors = [
    "Chocolate", "Vanilla", "Strawberry", "Cookies & Cream", "Banana",
    "Mango", "Peanut Butter", "Coconut", "Blueberry", "Caramel",
    "Coffee", "Matcha", "Cinnamon", "Hazelnut", "Mint",
    "Raspberry", "Pineapple", "Lemon", "Orange", "Gingerbread",
    "Choco-Mint", "Apple", "Peach", "Pistachio", "Watermelon",
    "Tropical Punch", "Berry Mix", "Coconut-Chocolate", "Grape", "Vanilla-Caramel"
]


# Генерация датасета
dataset = []
for i in range(30):
    item = {
        "product": f"Product {i + 1}",
        "type": "Gainer" if random.random() > 0.5 else "Protein",
        "is_vegan": random.choice([True, False]),
        "price_rub": random.randint(1000, 5000),
        "weight_g": random.randint(500, 2000),
        "flavor": random.choice(flavors)
    }
    dataset.append(item)

# Сохранение датасета в json файл
dataset_path = 'data/protein_dataset.json'
with open(dataset_path, 'w') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

#dataset_path
