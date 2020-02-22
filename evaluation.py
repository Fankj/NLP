from collections import Counter

class Eval:
    def __init__(self, gold, pred):
        assert len(gold)==len(pred)
        self.gold = gold
        self.pred = pred
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']

    def accuracy(self):
        numer = sum(1 for p,g in zip(self.pred,self.gold) if p==g)
        return numer / len(self.gold)

    def confusion_matrix(self):
        conf_mtr = {l: Counter() for l in self.CLASSES}
        for l in conf_mtr:
            for m in self.CLASSES:
                conf_mtr[l][m] = sum(1 for p,g in zip(self.pred,self.gold) if p==m and g==l)
        return conf_mtr

    def precision(self):
        prec = {l: 0 for l in self.CLASSES}
        for l in prec:
            prec[l] = sum(1 for p,g in zip(self.pred,self.gold) if p==g and p==l) / sum(1 for p in self.pred if p==l)
        return prec

    def recall(self):
        rec = {l: 0 for l in self.CLASSES}
        for l in rec:
            rec[l] = sum(1 for p,g in zip(self.pred,self.gold) if p==g and p==l) / sum(1 for p in self.gold if p==l)
        return rec

    def F1(self):
        f1 = {l: 0 for l in self.CLASSES}
        prec = self.precision()
        rec = self.recall()
        for l in f1:
            f1[l] = 2*prec[l]*rec[l]/(prec[l]+rec[l])
        return f1

