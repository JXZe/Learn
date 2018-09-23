import numpy as np
import copy

def embedding_lookup(w,i):
    h=w[i]
    return h

def embedding_lookup_djdy(djdh,w,i):
    w_new=np.zeros_like(w)
    w_new[i]=djdh
    return w_new


def linear_transform(x,linear_w,linear_b):
    h=np.dot(x,linear_w)+linear_b
    return h

def linear_transform_djdy(djdh,x,w,b):
    djdw=np.dot(x.reshape(1,-1).T,djdh.reshape(1,-1))
    djdb=djdh
    djdy=np.dot(djdh,w.T)
    return djdy,djdw,djdb



def Tanh(x):
    h=np.tanh(x)
    return h

def Tanh_djdy(djdy,x):
    djdx=djdy*(1-np.dot(x,x.T))
    return djdx


def Softmax(x):
    e=np.exp(x)
    e_sum=np.sum(e)
    h=e/e_sum
    return h

def cross_entropy(x,i):
    loss=-np.log(x[i])
    return loss

def softmax_djdy(x,i):
    djdy=x
    djdy[i]=(-1)/djdy[i]
    return djdy

def forward(embedding_w,preword,linear_w,linear_b,output_embeded_w,output_embeded_b,outputword):
   
    print('-------前向传播-------')
    h1=embedding_lookup(embedding_w,preword)
    print('h1=',h1)
    h2=linear_transform(h1,linear_w,linear_b)
    print('h2=',h2)
    h3=Tanh(h2)
    print('h3=',h3)
    h4=linear_transform(h3,output_embeded_w,output_embeded_b)
    print('h4=',h4)
    h5=Softmax(h4)
    print('h5=',h5)
    loss=cross_entropy(h5,outputword)
    print('')
    print('Loss=',loss)
    print('')
    
    return h1,h2,h3,h4,h5

def backword(h5,outputword,output_embeded_w,output_embeded_b,h3,h2,h1,linear_w,linear_b,embedding_w,preword):
   
    print('-------反向传递-------')
    djdh4=softmax_djdy(h5,outputword)
    print('djdh4=',djdh4)
    djdh3,djd_output_embeded_w,djd_output_embeded_b=linear_transform_djdy(djdh4,h3,output_embeded_w,output_embeded_b)
    print('djdh3=',djdh3)
    djdh2=Tanh_djdy(djdh3,h3)
    print('djdh2=',djdh2)
    djdh1,djd_linear_w,djd_linear_b=linear_transform_djdy(djdh2,h1,linear_w,linear_b)
    print('djdh1=',djdh1)
    djd_input_embedding=embedding_lookup_djdy(djdh1,embedding_w,preword)
    print('djd_input_embedding=',djd_input_embedding)
    
    return djd_output_embeded_w,djd_output_embeded_b,djd_linear_w,djd_linear_b,djd_input_embedding

def weight_updating(eta,embedding_w,djd_input_embedding,linear_w,djd_linear_w,linear_b,djd_linear_b,output_embeded_w,output_embeded_b,djd_output_embeded_w,djd_output_embeded_b):
    #print('------参数更新------')
    
    embedding_w += -eta * djd_input_embedding
    
    linear_w += -eta * djd_linear_w
    linear_b += -eta* djd_linear_b

    output_embeded_w += -eta * djd_output_embeded_w
    output_embeded_b += -eta*djd_output_embeded_b
    
    return embedding_w,linear_w,linear_b,output_embeded_w,output_embeded_b


def main():
    preword=0
    outputword=1
    
    embedding_w=np.array([[0.4,1],[0.2,0.4],[-0.3,2]])
    
    linear_w=np.array([[1.2,0.2],[-0.4,0.4]])
    linear_b=np.array([0,0.5])
    
    output_embeded_w=np.array([[-1,0.4,-0.3],[1,0.5,0.2]])
    output_embeded_b=np.array([0,0.5,0])
    
    eta=0.1
    
    for i in range(10):
        print("======================")
        print("      迭代第%d次     " %i)
        print("======================")
        print()
        h1,h2,h3,h4,h5=forward(embedding_w,preword,linear_w,linear_b,output_embeded_w,output_embeded_b,outputword)
        print()
        label_end=copy.deepcopy(h5)
        djd_output_embeded_w,djd_output_embeded_b,djd_linear_w,djd_linear_b,djd_input_embedding=backword(h5,outputword,output_embeded_w,output_embeded_b,h3,h2,h1,linear_w,linear_b,embedding_w,preword)
        print()
        embedding_w,linear_w,linear_b,output_embeded_w,output_embeded_b=weight_updating(eta,embedding_w,djd_input_embedding,linear_w,djd_linear_w,linear_b,djd_linear_b,output_embeded_w,output_embeded_b,djd_output_embeded_w,djd_output_embeded_b)
        print()
    print("======================")
    print('       最终结果        ')
    print("======================")
   
    a=label_end[0]
    b=label_end[1]
    c=label_end[2]
    
    if np.max([a,b,c])==a:
        print('下一个字母为a')
    elif np.max([a,b,c])==b:
        print('下一个字母为b')
    elif np.max([a,b,c])==c:
        print('下一个字母为c')  
    else:
        print('模型失效，请联系工程人员')
    
if __name__=="__main__":
    main()
