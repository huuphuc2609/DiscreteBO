class Space:
    def __init__(self, lower, upper, inIsUniformSpaced=True):
        self.lower = lower
        self.upper = upper
        self.isUniformlySpaced = inIsUniformSpaced
        self.valueSet = []

    # If values are not uniformly spaced then we need to define
    def setValuesSet(self, arrValues):
        self.valueSet = arrValues

    def getCeil(self, val):
        result = val
        if val in self.valueSet:
            return val
        for x in self.valueSet:
            if x > val:
                result = x
                break
        return result

    def getFloor(self, val):
        result = val
        if val in self.valueSet:
            return val
        for x in self.valueSet:
            if x < val:
                result = x
            else:
                break
        return result