from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel



def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('./dataset.csv','r'), delimiter=',', dtype='f8')[1:]
    dataset2 = genfromtxt(open('./dataset.csv','r'), delimiter=',', dtype=None)[1:]
    #print dataset 
    target = [x[len(x)-1] for x in dataset2]
    train = [x[0:len(x)-2] for x in dataset]
    print len(train[1])
    clf = RandomForestClassifier(n_estimators=10)
    d = clf.fit(train, target)
    clf = ExtraTreesClassifier()
    clf = clf.fit(train, target)
    clf.feature_importances_  
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(train)
    print X_new.shape  
    print X_new           
    #train_new = d.transform(train)
    #testin = genfromtxt(open('./test.csv','r'), delimiter=',', dtype='f8')[1:]
    #test = [x[1:93] for x in testin]
    #test_new = d.transform(test)
    #create and train the random forest
    #multi-core CPUs can use: estimator = 400 gave best result 2224 	
	
    #rf = RandomForestClassifier(n_estimators=300)
    #rf = RandomForestClassifier(n_estimators=100)
    #rf.fit(train, target)

    #savetxt('submission8.csv', rf.predict_proba(test), delimiter=',', fmt='%s')

if __name__=="__main__":
    main()
