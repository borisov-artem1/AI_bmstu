import re

from commands import *
from tmp import type_dict
from prepareData import initPrefer, resetPrefer
from preprocessing import preprocessing, updateType, updateIntensity
from rules import *


def _getAnswer():
    while True:
        answer = input().lower()
        if answer == "yes":
            return True
        elif answer == "no":
            return False

        cmdYesNoValidation()


def isFound():
    cmdWasFound()
    return _getAnswer()


def isAdd():
    cmdAddDefinition()
    return _getAnswer()


def isReset():
    cmdResetDefinition()
    return _getAnswer()


def processDefinition(dictPrefer, data):
    f = 0
    for rule in RULE_ARR:
        regexp = re.compile(rule)
        match = regexp.match(data)
        if match is not None:
            resDict = match.groupdict()

            if rule == NOT_SIMILAR_TO_BRAND or rule == I_DISLIKE_BRAND:
                dictPrefer["dislikes"].append(resDict["similar_name"])
            elif rule == SIMILAR_TO_BRAND or rule == I_LIKE_BRAND:
                dictPrefer["likes"].append(resDict["similar_name"])
            elif rule == WANT_ABSTRACT_OBJ:
                dictPrefer["flavor"].append(resDict["obj"])
            elif rule == WANT_ABSTRACT_OBJ_KINDPARFUM:
                dictPrefer["flavor"].append(resDict["obj"])

                if resDict["kind_sport_nutrition"] is not None:
                    dictPrefer["type"].append(updateType(resDict["kind_sport_nutrition"]))
            elif rule == WANT_ABSTRACT:
                dictPrefer["flavor"].append(resDict["tag1"])
            elif rule == WHAT_EXISTS_KINDPARFUM:
                dictPrefer["type"].append(updateType(resDict["kind_sport_nutrition"]))
            elif rule == I_LIKE_TAG:
                dictPrefer["flavor"].append(resDict["tag1"])
            elif rule == I_LIKE_OBJ:
                dictPrefer["flavor"].append(resDict["obj"])
            elif rule == COUNTRY_EXT_KINDPARFUM_1 or rule == COUNTRY_EXT_KINDPARFUM_2:
                if resDict["country"] is not None:
                    dictPrefer["country"].append(resDict["country"])
                if resDict["country_ext"] is not None:
                    dictPrefer["country"].append(resDict["country_ext"])

                dictPrefer["type"].append(updateType(resDict["kind_sport_nutrition"]))
            elif rule == COUNTRY_EXT:
                if resDict["country"] is not None:
                    dictPrefer["country"].append(resDict["country"])
                if resDict["country_ext"] is not None:
                    dictPrefer["country"].append(resDict["country_ext"])
            elif rule == SHOW_DURABILITY:
                dictPrefer["intensity"].append(updateIntensity(resDict["intensity"]))

            f = 1
            break

    if f == 0:
        cmdMissunderstanding()
    cmdFind(dictPrefer)


def dialog():
    dictPrefer = initPrefer()

    while True:
        cmdDescribe()
        data = input()
        dataProcessed = preprocessing(data)
        processDefinition(dictPrefer, dataProcessed)

        while True:
            if isFound():
                cmdGoodBye()
                return
            elif isAdd():
                break
            elif isReset():
                resetPrefer(dictPrefer)
                cmdResetDefinitionComplete()
                break


def main():
    cmdWelcome()
    dialog()


if __name__ == "__main__":
    main()
