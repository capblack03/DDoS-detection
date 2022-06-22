import pandas as pd
import numpy as np
from sklearn.model_selection import tarin_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import arff as arf

from google.colab import files
file = files.upload()

decoder = arf.ArffDecoder()
data = decoder.decode(file,encode_noinal=True)

vals = [val[0:-1] for val in data['data']]
labels = [lab[-1] for lab in data['data']]

da = set(labels)
brac = 600
temp1= []
tempd = []
for i in da:
    count = 0
    while count < brac:
        for j in range(len(labels)):
            if labels[j]:
                temp1.append(label[j])
                tempd.append(vals[j])
                count+=1
            if count==brac:
                break
vals = tempd
labels = temp1

l = len(vals)
print(l)

X_train,X_test,Y_train,Y_test = train_test_split(vals,labels,stratify=labels,test_size=0.2,random_state=0)

Scaler = StandardScaler()
x_train = Scaler.fit_transform(X_train)
x_test = Scaler.transform(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

model1 = SVC(kernel='sigmoid',gamma='auto')
model1.fit(x_train,y_train)

y_pred1 = model.predict(x_test)

print((accuracy_score(y_pred1,y_test))*100,"%")

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(x_train,y_train)

y_pred2 = model2.predict(x_test)

print((accuracy_score(y_pred2,y_test))*100,"%")

model3 = GaussianNB()
model3.fit(x_train,y_train)

y_pred2 = model2.predict(x_test)

print((accuracy_score(y_pred2,y_test))*100,"%")

train_x,val_x,train_y,val_y = train_test_split(x_tarin,y_train,stratify=y_train,test_size=0.2,random_state=0)

print(x_train.shape,x_test.shape)

columns = ['SRC_ADD','DES_ADD','PKT_ID','FROM_NODE','TO+NODE','PKT_TYPE',
            'PKT_SIZE','FLAGS','FID','SEQ_NUMBER','NUMBER_OF_PKT',
            'NUMBER_OF_BYTE','NODE_NAME_FROM','NODE_NAME_TO','PKT_IN','PKT_OUT'
            'PKT_R','PKT_DELAY_NODE','PKT_RATE','BYTE_RATE','PKT_AVG_SIZE',
            'UTILIZATION','PKT_DELAY','PKT_SEND_TIME','PKT_RESERVED_TIME',
            'FIRST_PKT_SENT','LAST_PKT_RESERVED']


model1 = SVC(kernel='sigmoid',gamma='auto')
model1.fit(train_x,train_y)
y_val_pred1 = model1.predict(val_x)
y_val_pred1 = pd.DataFrame(y_val_pred1)
y_test_pred1 = model1.predict(x_test)
y_test_pred1 = pd.DataFrame(y_test_pred1)

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_x,train_y)
y_val_pred2 = model2.predict(val_x)
y_val_pred2 = pd.DataFrame(y_val_pred2)
y_test_pred2 = model2.predict(x_test)
y_test_pred2 = pd.DataFrame(y_test_pred2)

model3 = GaussianNB()
model3.fit(train_x,train_y)
y_val_pred3 = model3.predict(val_x)
y_val_pred3 = pd.DataFrame(y_val_pred3)
y_test_pred3 = model3.predict(x_test)
y_test_pred3 = pd.DataFrame(y_test_pred3)

val_input = pd.concat([pd.DataFrame(val_x,columns=columns),y_val_pred1,y_val_pred2,y_val_pred3],axis=1)
test_input = pd.concat([pd.DataFrame(x_test,columns=columns),y_test_pred1,y_test_pred2,y_test_pred3],axis=1)

model = RandomForestClassifier(n_estimators=200)
mode.fit(val_input,val_y)

print((model.score(test_input,y_test))*100,"%")