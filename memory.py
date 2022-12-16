class Memory:
    def __init__(self, kw_list) -> None:
        self.kw_list = kw_list
        self.kw_len = len(kw_list)
        self.process = 0
        self.history = ""
    
    def update(self):
        self.process += 1
    
    def not_end(self):
        if self.process < self.kw_len:
            return True
        else:
            return False
    
    def history_update(self, s):
        self.history = self.history + s