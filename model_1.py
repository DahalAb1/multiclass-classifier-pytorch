from sklearn.model_selection import train_test_split
import torch 
from sklearn.datasets import make_circles
from torch import nn

circle_plts = 1000 
X,Y = make_circles(circle_plts,noise = 0.03, random_state=42)

#change data into tensors, do not forget this 
X = torch.from_numpy(X).type(torch.float)
Y = torch.from_numpy(Y).type(torch.float)


class CircleModelVO(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,out_features=10)
        self.layer_2 = nn.Linear(in_features=10,out_features=10)
        self.layer_3 = nn.Linear(in_features=10,out_features=1)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model_1 = CircleModelVO()

loss_Function = nn.BCELoss()
optimizer = torch.optim.SGD(model_1.parameters(),lr=0.1)

def accuracy_fn(y_true, y_pred):
    total = len(y_true)
    correct = torch.eq(y_true,y_pred).sum().item() 
    accuracy = (correct/total)*100

    return accuracy



torch.manual_seed(42)

epochs = 2000


for epoch in range(epochs):
    #training mode 
    model_1.train() 

    #forward pass 
    y_pred = model_1(X_train).squeeze()

    #calculate the loss 
    logits = (torch.sigmoid(y_pred))
    prob_pred = torch.round(logits)

    #using BCELoss will work on raw logits 
    loss = loss_Function(logits,y_train)
    acc = accuracy_fn(y_true=y_train,y_pred=prob_pred)

    #zero grad 
    optimizer.zero_grad()

    #backpropogation 
    #caculate loss of gradients just predicted 
    loss.backward()

    #optimizer step 
    optimizer.step()

    #testing model 
    model_1.eval()

    with torch.inference_mode():
        test_prediction = model_1(X_test).squeeze()
        prob_test = torch.sigmoid(test_prediction)
        test_loss = loss_Function(prob_test,y_test)
        test_pred_binary = torch.round(prob_test)
        test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred_binary)

    if epoch % 100 == 0:
        print(
            f'Epoch: {epoch}',
             f'training accuracy: {acc:.2f} and training loss: {loss:.2f}',
            f'testing accuracy: {test_acc:.2f} and testing loss: {test_loss:.2f}'
        )





