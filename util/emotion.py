class Emotion:
    def __init__(self):
        pass

    HAPPINESS = 0
    HOPE = 1
    NEUTRALITY = 2
    SADNESS = 3
    ANGER = 4
    ANXIETY = 5
    TIREDNESS = 6
    REGRET = 7

    def to_string(self, num):
        if num == self.HAPPINESS:
            return "기쁨"
        if num == self.HOPE:
            return "희망"
        if num == self.NEUTRALITY:
            return "행복"
        if num == self.ANGER:
            return "분노"
        if num == self.SADNESS:
            return "슬픔"
        if num == self.ANXIETY:
            return "불안"
        if num == self.TIREDNESS:
            return "피곤"
        if num == self.REGRET:
            return "후회"
