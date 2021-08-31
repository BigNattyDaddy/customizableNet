import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#input is a single number and so is output



class network:

    def __init__(self, neuronsIn, layers, inputNum, outputNum):

        self.Yh = np.zeros(neuronsIn)

        self.L = layers
        self.neur = neuronsIn #neurons in is number of neurons for in between layers
        self.inputNum = inputNum
        self.outputNum = outputNum


        self.ch = {}
        self.param = {}
        self.backE = {}

        self.lr = 0.001


    def initialize(self):

        for i in range (self.L):

            #for input layer initialization
            if(i == 0):
                self.param['w' + str(i)] = np.random.randn(self.neur, 1) / 10*np.sqrt(1)

                self.param['b'+str(i)] = np.zeros((self.neur,1))


            #for output layer initialization
            # elif (i == self.L-1):
            #     self.param['w'+str(i)] = np.random.randn(1,self.neur)/np.sqrt(self.outputNum)
            #
            #     self.param['b' + str(i)] = np.zeros((self.neur,1 ))


            else:

                self.param['w' + str(i)] = np.random.randn(self.neur, self.neur) / np.sqrt(self.neur)

                self.param['b' + str(i)] = np.zeros((self.neur, 1))


    def forward(self, input):

        for i in range(self.L):

            if(i==0):

                z = self.param['w'+str(i)].dot(input) + self.param['b'+str(i)]

                a = Relu(z)
                self.ch['z'+str(i)], self.ch['a'+str(i)]=z,a


            else:

                z = self.param['w' + str(i)].dot(self.ch['a'+str(i-1)]) + self.param['b' + str(i)]

                # if(i==self.L-1):
                #        a = Relu(z)

                a = Relu(z)

                #this might cause output to be zero. If so, change this so that last layer activation is sigmoid and fix it also in backprop

                self.ch['z' + str(i)], self.ch['a' + str(i)] = z, a

                if(i == self.L-1):
                    self.backE['Yh']=np.sum(self.ch['a'+str(i)])



                #possibility of changing the activation function here

        return self.backE['Yh']


    def backwards(self, input, expect):

        #derivative of loss from output-----might be incorrect, but trying this way to avoid NaN
        self.backE['dL_Yh'] = -(expect - self.backE['Yh'])

        for i in range(self.L):

            if (i == 0):

                dl_z = self.backE['dL_Yh'] * dRelu(self.ch['z'+str(self.L-1-i)])

                self.backE['dl_a'+str(self.L-i-1)] = np.dot(self.param['w'+str(self.L-i-1)].T, dl_z)

                dl_weight = 1./self.neur*np.dot(dl_z, self.ch['a'+str(self.L-i-1)].T)

                dl_bias = 1./self.neur*np.dot(dl_z, np.ones([dl_z.shape[1],1]))


            elif (i == self.L - 1):

                #activation from previous layer in backwards process is used here(thats why not self.L-i-1 just -i
                dl_z = self.backE['dl_a' + str(self.L - i )] * dRelu(self.ch['z' + str(self.L - i-1)])

                self.backE['dl_a'+str(self.L-i-1)] = np.dot(self.param['w' + str(self.L - i-1)].T, dl_z)

                dl_weight = np.dot(dl_z, input)

                dl_bias = np.dot(dl_z, np.ones([dl_z.shape[1], 1]))


            else:

                dl_z = self.backE['dl_a'+str(self.L-i)] * dRelu(self.ch['z' + str(self.L - i-1)])

                self.backE['dl_a'+str(self.L-i-1)] = np.dot(self.param['w' + str(self.L - i-1)].T, dl_z)

                dl_weight = 1. / self.neur * np.dot(dl_z, self.ch['a' + str(self.L - i-1)].T)

                dl_bias = 1. / self.neur * np.dot(dl_z, np.ones([dl_z.shape[1], 1]))


            self.param['w'+str(self.L-1-i)] = self.param['w'+str(self.L-1-i)]-self.lr*dl_weight
            self.param['b' + str(self.L - 1 - i)] = self.param['b' + str(self.L - 1 - i)] - 0.001 * dl_bias



            print(self.param['b'+str(self.L-1-i)])














def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def Relu(Z):
    return np.maximum(-0.2*Z, Z)

def dRelu(x):
        x[x <= 0] = -0.2
        x[x > 0] = 1
        return x


def dSigmoid(Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = s * (1 - s)
        return dZ


x = np.arange(1,11,0.1)
y = 3*x
x,y = shuffle(x,y, random_state=0)



net = network(15,6,1,1)
net.initialize()

for i in range(100):

    Yh = net.forward(x[i])
    net.backwards(x[i], y[i])

newboi = np.arange(11,16,0.1)
newy = []

for i in range(50):
    Yh= net.forward(newboi[i])
    newy.append(Yh)






plt.figure()
plt.subplot(211)
plt.scatter(newboi, np.reshape(np.array(newy), (50,)))

plt.show()




