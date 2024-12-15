from tmp import namesUI, giveRecommendationFull, merged_df


def initPrefer():
    return {"likes": [], "dislikes": [], "type": [],
            "flavor": [], "country": [], "intensity": []}


def resetPrefer(dictPrefer):
    for key in dictPrefer.keys():
        dictPrefer[key] = []
    return dictPrefer


def _replaceBrand(inputArr):
    brandArr = []
    brandUI = {}
    for i in range(len(namesUI)):
        brandUI[namesUI[i].lower()] = i

    for brand in inputArr:
        curName = brand.lower()
        if curName in brandUI.keys():
            brandArr.append(namesUI[brandUI[curName]])
    return brandArr



def _replaceCountry(inputArr):
    countryArr = []
    countryDict = {"italy": "Italy", "france": "France", "usa": "USA", "russia": "Russia",
                   "united kingdom":"United Kingdom", "spanish":"Spanish", "poland":"Poland",
                   "hawaii":"Hawaii", "germany": "Germany","italian": "Italy", "russian": "Russia",
                   "german": "Germany", "french": "France", "american": "USA", "united states":"USA",
                   "united states of america": "USA","english": "United Kingdom", "polish": "Poland","hawaiian": "Hawaii"}

    for country in inputArr:
        curCountry = country.lower()
        if curCountry in countryDict.keys():
            countryArr.append(countryDict[curCountry])
    return countryArr


def find(paramDict):
    paramDict["likes"] = _replaceBrand(paramDict["likes"])
    paramDict["dislikes"] = _replaceBrand(paramDict["dislikes"])
    paramDict["country"] = _replaceCountry(paramDict["country"])
    return giveRecommendationFull(paramDict["intensity"], paramDict["country"], [],
                           paramDict["type"], [], paramDict["flavor"],
                           paramDict["likes"], paramDict["dislikes"], merged_df)
