class Positive_Negative:
    def __init__(self):
        pass

    POSITIVE = 0
    NEUTRAL = 1
    NEGATIVE = 2

    def to_string(self, num):
        if num == self.POSITIVE:
            return "긍정"
        if num == self.NEUTRAL:
            return "중립"
        if num == self.NEGATIVE:
            return "부정"
