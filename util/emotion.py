class Emotion:
    def __init__(self):
        pass

    JOY = 0
    HOPE = 1
    NEUTRALITY = 2
    SADNESS = 3
    ANGER = 4
    ANXIETY = 5
    TIREDNESS = 6
    REGRET = 7

    def to_string(self, num):
        if num == self.JOY:
            return "기쁨"
        if num == self.HOPE:
            return "희망"
        if num == self.NEUTRALITY:
            return "중립"
        if num == self.SADNESS:
            return "슬픔"
        if num == self.ANGER:
            return "분노"
        if num == self.ANXIETY:
            return "불안"
        if num == self.TIREDNESS:
            return "피곤"
        if num == self.REGRET:
            return "후회"
        
    def to_num(self, st):
        st = st.strip()
        if st == "기쁨":
            return self.JOY
        if st == "희망":
            return self.HOPE
        if st == "중립":
            return self.NEUTRALITY
        if st == "슬픔":
            return self.SADNESS
        if st == "분노":
            return self.ANGER
        if st == "불안":
            return self.ANXIETY
        if st == "피곤":
            return self.TIREDNESS
        if st == "후회":
            return self.REGRET
