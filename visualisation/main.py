from graphviz import Digraph
from IPython.display import Image

if __name__ == '__main__':
    flavor_tree = Digraph(comment='Flavor Tree', format='png')

    # Добавляем узлы для основной категории
    flavor_tree.node('F', 'Flavors')

    # Категории
    flavor_tree.node('F1', 'Fruity')
    flavor_tree.node('F2', 'Nutty')
    flavor_tree.node('F3', 'Dessert')
    flavor_tree.node('F4', 'Beverages')
    flavor_tree.node('F5', 'Spicy')
    flavor_tree.node('F6', 'None')


    # Соединяем категории с корнем
    flavor_tree.edge('F', 'F1')
    flavor_tree.edge('F', 'F2')
    flavor_tree.edge('F', 'F3')
    flavor_tree.edge('F', 'F4')
    flavor_tree.edge('F', 'F5')
    flavor_tree.edge('F', 'F6')

    # Подкатегории для Fruity
    flavor_tree.node('F11', 'Tropical Fruits')
    flavor_tree.node('F12', 'Citrus Fruits')
    flavor_tree.node('F13', 'Berries')
    flavor_tree.node('F14', 'Stone Fruits')

    flavor_tree.edge('F1', 'F11')
    flavor_tree.edge('F1', 'F12')
    flavor_tree.edge('F1', 'F13')
    flavor_tree.edge('F1', 'F14')

    # Вкусы для Tropical Fruits
    flavor_tree.node('F111', 'Banana')
    flavor_tree.node('F112', 'Mango')
    flavor_tree.node('F113', 'Pineapple')
    flavor_tree.node('F114', 'Coconut')
    flavor_tree.node('F115', 'Tropical Punch')

    flavor_tree.edge('F11', 'F111')
    flavor_tree.edge('F11', 'F112')
    flavor_tree.edge('F11', 'F113')
    flavor_tree.edge('F11', 'F114')
    flavor_tree.edge('F11', 'F115')

    # Вкусы для Citrus Fruits
    flavor_tree.node('F121', 'Lemon')
    flavor_tree.node('F122', 'Orange')

    flavor_tree.edge('F12', 'F121')
    flavor_tree.edge('F12', 'F122')

    # Вкусы для Berries
    flavor_tree.node('F131', 'Strawberry')
    flavor_tree.node('F132', 'Blueberry')
    flavor_tree.node('F133', 'Raspberry')
    flavor_tree.node('F134', 'Berry Mix')
    flavor_tree.node('F135', 'Grape')

    flavor_tree.edge('F13', 'F131')
    flavor_tree.edge('F13', 'F132')
    flavor_tree.edge('F13', 'F133')
    flavor_tree.edge('F13', 'F134')
    flavor_tree.edge('F13', 'F135')

    # Вкусы для Stone Fruits
    flavor_tree.node('F141', 'Peach')
    flavor_tree.node('F142', 'Watermelon')
    flavor_tree.node('F143', 'Apple')

    flavor_tree.edge('F14', 'F141')
    flavor_tree.edge('F14', 'F142')
    flavor_tree.edge('F14', 'F143')

    # Вкусы для Nutty
    flavor_tree.node('F21', 'Peanut Butter')
    flavor_tree.node('F22', 'Hazelnut')
    flavor_tree.node('F23', 'Pistachio')

    flavor_tree.edge('F2', 'F21')
    flavor_tree.edge('F2', 'F22')
    flavor_tree.edge('F2', 'F23')

    # Вкусы для Dessert
    flavor_tree.node('F31', 'Chocolate-based')
    flavor_tree.node('F32', 'Vanilla-based')
    flavor_tree.node('F33', 'Caramel-based')
    flavor_tree.node('F34', 'Other Desserts')

    flavor_tree.edge('F3', 'F31')
    flavor_tree.edge('F3', 'F32')
    flavor_tree.edge('F3', 'F33')
    flavor_tree.edge('F3', 'F34')

    # Вкусы для Chocolate-based
    flavor_tree.node('F311', 'Chocolate')
    flavor_tree.node('F312', 'Cookies & Cream')
    flavor_tree.node('F313', 'Choco-Mint')
    flavor_tree.node('F314', 'Coconut-Chocolate')

    flavor_tree.edge('F31', 'F311')
    flavor_tree.edge('F31', 'F312')
    flavor_tree.edge('F31', 'F313')
    flavor_tree.edge('F31', 'F314')

    # Вкусы для Vanilla-based
    flavor_tree.node('F321', 'Vanilla')
    flavor_tree.node('F322', 'Vanilla-Caramel')

    flavor_tree.edge('F32', 'F321')
    flavor_tree.edge('F32', 'F322')

    # Вкусы для Caramel-based
    flavor_tree.node('F331', 'Caramel')
    flavor_tree.node('F332', 'Vanilla-Caramel')

    flavor_tree.edge('F33', 'F331')
    flavor_tree.edge('F33', 'F332')

    # Вкусы для Other Desserts
    flavor_tree.node('F341', 'Gingerbread')

    flavor_tree.edge('F34', 'F341')

    # Вкусы для Beverages
    flavor_tree.node('F41', 'Coffee-based')
    flavor_tree.node('F42', 'Tea-based')

    flavor_tree.edge('F4', 'F41')
    flavor_tree.edge('F4', 'F42')

    # Вкусы для Coffee-based
    flavor_tree.node('F411', 'Coffee')

    flavor_tree.edge('F41', 'F411')

    # Вкусы для Tea-based
    flavor_tree.node('F421', 'Matcha')

    flavor_tree.edge('F42', 'F421')

    # Вкусы для Spicy
    flavor_tree.node('F51', 'Cinnamon')
    flavor_tree.node('F52', 'Mint')
    flavor_tree.node('F53', 'Choco-Mint')

    flavor_tree.edge('F5', 'F51')
    flavor_tree.edge('F5', 'F52')
    flavor_tree.edge('F5', 'F53')



    # Финализация и рендер дерева
    flavor_tree.render('flavor_tree', format='png')  # Сохраняем дерево в файл
    Image(filename='data/flavor_tree.png')  # Отображаем изображение