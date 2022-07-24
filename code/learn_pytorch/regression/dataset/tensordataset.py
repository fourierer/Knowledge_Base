# 创建适用于加载当前数据的类
class TensorDataset(object):
    def __init__(self, features, labels):
        assert features.size(0) == labels.size(0)
        self.features = features
        self.labels = labels
    
    def __getitem__(self,index):
        return (self.features[index],self.labels[index])
    
    def __len__(self):
        return len(self.features)