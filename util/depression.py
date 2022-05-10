class Depression:
    def __init__(self):
        pass

    NO_DEPRESS = 0
    DEPRESS = 1

    def to_string(self, num):
        if num == self.NO_DEPRESS:
            return "비우울"
        if num == self.DEPRESS:
            return "우울"

    def to_num(self, st):
        st = st.strip()
        if st == "비우울":
            return self.NO_DEPRESS
        if st == "우울":
            return self.DEPRESS
