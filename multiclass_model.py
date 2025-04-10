from torch import nn 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import torch 


RANDOM_STATE = 42 

X,y = make_blobs(1000,n_features=2,centers=4,cluster_std=1,random_state=RANDOM_STATE) 


X_blob = torch.from_numpy(X).type(torch.float)
y_blob = torch.from_numpy(y)

def showData():
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)
    plt.show() 

def accuracy_fn(y_true, y_pred):
    total = len(y_true)
    correct = torch.eq(y_true,y_pred).sum().item() 
    accuracy = (correct/total)*100

    return accuracy

#lets create a model 

class MultiDef(nn.Module):
    def __init__(self,input_features, output_features,hiddel_units = 8):
        super().__init__()

        self.layer1 = nn.Linear(in_features=input_features,out_features=hiddel_units)
        self.layer2 = nn.Linear(in_features=hiddel_units,out_features=hiddel_units)
        self.layer3 = nn.Linear(in_features=hiddel_units,out_features=output_features)
        relu = nn.ReLU()

        # self.linear_layer = nn.Sequential(
        #     # nn.ReLU(),
        #     nn.Linear(in_features=input_features,out_features=hiddel_units),
        #     # nn.ReLU(),
        #     nn.Linear(in_features = hiddel_units, out_features=hiddel_units),
        #     # nn.ReLU(),
        #     nn.Linear(in_features=hiddel_units,out_features=output_features)
        # )

    def forward(self,x):
        return self.layer3(self.layer2(self.layer1(x)))
    
class_model_1 = MultiDef(input_features=2,output_features=4)

X_train,X_test,y_train,y_test = train_test_split(X_blob,y_blob,test_size=0.2,random_state=RANDOM_STATE)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(class_model_1.parameters(),lr=0.01)

epocs = 120

for epoc in range(epocs):
    class_model_1.train()
    y_pred = class_model_1(X_train)
    prob_pred =torch.softmax(y_pred,dim=1).argmax(dim=1)
    trainloss = loss_function(y_pred,y_train)
    optimizer.zero_grad()
    trainloss.backward()
    optimizer.step()
    
    acc = accuracy_fn(y_true=y_train,y_pred=prob_pred)


    class_model_1.eval()
    with torch.inference_mode():
        y_test_pred = class_model_1(X_test)
        testloss = loss_function(y_test_pred,y_test)
        test_prob_pred = torch.softmax(y_test_pred,dim=1).argmax(dim=1)
        test_acc = accuracy_fn(y_true=y_test,y_pred=test_prob_pred)

    if epoc % 10 == 0:
        print(f"Epoch [{epoc}]")
        print(f"  Train Loss: {trainloss:.4f} | Train Accuracy: {acc:.2f}%")
        print(f"  Test  Loss: {testloss:.4f} | Test  Accuracy: {test_acc:.2f}%")
        print("-" * 50)


