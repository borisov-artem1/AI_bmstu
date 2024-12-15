from tmp import product_df
from params import *
from prepareData import find


def cmdOffer():
    print(DEFAULT_DATA)


def cmdWelcome():
    print(WELCOME_PHRASE)


def cmdGoodBye():
    print(GOODBYE_PHRASE)


def cmdDescribe():
    print(DESCRIBE)


def cmdDefault():
    cmdOffer()
    print(product_df.loc[1:5, ["brand", "country", "type", "is_vegan", "intensity"]])
    print()


def cmdWasFound():
    print(FOUND_QUESTION)


def cmdYesNoValidation():
    print(YES_NO)


def cmdAddDefinition():
    print(ADD_DEFINITION)


def cmdResetDefinition():
    print(RESET_PHRASE)


def cmdResetDefinitionComplete():
    print(RESET_PHRASE_COMPLETE)


def cmdMissunderstanding():
    print(MISUNDERSTANDING)


def cmdGiveMustRecomendation():
    print(MUST_LIKE)


def cmdGiveMayRecomendation():
    print(MAY_LIKE)


def _printRecomendations(recArr):
    iArr = []
    n = min(len(recArr), 5)
    for i in range(n):
        iArr.append(product_df.index[product_df["brand"] == recArr[i]].tolist()[0])
    print(product_df.loc[iArr, ["brand", "country", "type", "is_vegan", "intensity"]])


def cmdFind(dictPrefer):
    recMust, recMaybe = find(dictPrefer)

    if len(recMust):
        cmdGiveMustRecomendation()
        _printRecomendations(recMust)

    if len(recMaybe):
        cmdGiveMayRecomendation()
        _printRecomendations(recMaybe)
