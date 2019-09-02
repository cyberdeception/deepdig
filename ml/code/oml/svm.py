data_path = '/home/debo/Downloads/CWE119/newdata'

data = pd.read_csv(data_path, header=None).values
print("Data Reading Done")
n, _ = data.shape

# randomly split
idx = np.arange(len(data))
np.random.shuffle(idx)

train_num = int(0.5 * n)
valid_num = int(0.1 * n)

train_idx = idx[:train_num]
valid_idx = idx[train_num:train_num+valid_num]
test_idx = idx[train_num+valid_num:]

train_data = data[train_idx]
valid_data = data[valid_idx]
test_data = data[test_idx]

# get features and labels
train_feature = train_data[:, :-1]
train_label = train_data[:, -1]

valid_feature = valid_data[:, :-1]
valid_label = valid_data[:, -1]

test_feature = test_data[:, :-1]
test_label = test_data[:, -1]


from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(train_feature, train_label)  


y_pred = svclassifier.predict(test_feature) 

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(test_label,y_pred))  
print(classification_report(test_label,y_pred))   



