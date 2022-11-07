import os

class DataLoader:
    def __init__(self):
        self.keyword_list = []
        
    def readfile(self, dir):
        filepath = os.path.join(dir, 'keywords.txt')
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                kws = [k.strip() for k in line.strip().split(',')]
                self.keyword_list.append(kws)
    