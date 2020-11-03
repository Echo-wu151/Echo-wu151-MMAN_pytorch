import numpy as np
import pdb
import sys
import matplotlib.pyplot as plt
import os
def loadMetrics(folderName):
    # Loss
    loss = np.load(folderName + os.sep+folderName+'_loss.npy')
    dice = np.load(folderName + os.sep+folderName+'_DSCs.npy')


    # Dice training

    
    return loss,dice

def save_result(result1,result2, result_type, save_path, record):
    x = np.arange(0, len(result1))
    # y = np.array(result1)
    y1 = np.array(result1)
    y2 = np.array(result2)
    # y1 = y[:, 0]
    # y2 = y[:, 1]
    plt.plot(x, y1, color="r", marker="o")
    plt.plot(x, y2, color="b", marker="*")
    plt.xlabel("epoch(s)")
    if result_type == "loss":
        plt.ylabel("loss")
    elif result_type == "acc":
        plt.ylabel("acc")
    else:
        raise ValueError('Not support result_type')
    plt.legend(["%s_train" % result_type, "%s_test" % result_type])
    plt.savefig("%s/%s_%s" % (save_path, record, result_type), dpi=120)
    plt.close()
    # plt.show()

def plot1Models(modelNames):
    model1Name = modelNames[0]


    [loss1, DSC1] = loadMetrics(model1Name)


    numEpochs1 = len(loss1)


    lim = numEpochs1
    print('lim',lim)
    print(len(DSC1))
    # if numEpochs2 < numEpochs1:
    #     lim = numEpochs2

    # Plot features
    xAxis = np.arange(0, lim, 1)
    # xAxis = np.arange(0, 370, 10)

    plt.figure(1)

    # Training Dice
    # plt.subplot(212)

    # plt.plot(xAxis, DSC1[0:lim].mean(axis=2), 'r-', label=model1Name, linewidth=2)
    plt.plot(xAxis, loss1[0:lim].mean(axis=2), 'b.', label=model1Name, linewidth=2)
    legend = plt.legend(loc='lower center', shadow=True, fontsize='large')
    # legend = plt.legend(loc='lower center', shadow=True, fontsize='large')
    plt.title('DSC Validation)')
    plt.grid(True)
    plt.ylim([0.0, 1])
    plt.xlabel('Number of epochs')
    plt.ylabel('DSC')
    # pdb.set_trace()
    # plt.xlim([0, 10,370])

    plt.show()



def plot2Models(modelNames):

    model1Name = modelNames[0]
    model2Name = modelNames[1]
    
    [loss1, DSC1] = loadMetrics(model1Name)
    [loss2, DSC2] = loadMetrics(model2Name)
    
    numEpochs1 = len(loss1)
    numEpochs2 = len(loss2)
    
    lim = numEpochs1
    if numEpochs2 < numEpochs1:
        lim = numEpochs2
        

    # Plot features
    #xAxis = np.arange(0, lim, 1)
    xAxis = np.arange(0, 370, 10)

    plt.figure(1)

    # Training Dice
    #plt.subplot(212)

    plt.plot(xAxis, DSC1[0:lim].mean(axis=2), 'r-', label=model1Name,linewidth=2)
    plt.plot(xAxis, DSC2[0:lim].mean(axis=2), 'b-', label=model2Name,linewidth=2)
    legend = plt.legend(loc='lower center', shadow=True, fontsize='large')
    plt.title('DSC Validation)')
    plt.grid(True)
    plt.ylim([0.0, 1])
    plt.xlabel('Number of epochs')
    plt.ylabel('DSC')
    #pdb.set_trace()
    #plt.xlim([0, 10,370])

    plt.show()


def plot(argv):

    modelNames = []
    
    numModels = len(argv)
    
    for i in range(numModels):
        modelNames.append(argv[i])
    
    def oneModel():
        print ("-- Ploting one model --")
        plot1Models(modelNames)

    def twoModels():
        print ("-- Ploting two models --")
        plot2Models(modelNames)
        
    def threeModels():
        print ("-- Ploting three models --")
        plot3Models(modelNames)
        
    def fourModels():
        print ("-- Ploting four models --")
        plot4Models(modelNames)
        
    # map the inputs to the function blocks
    options = {1 : oneModel,
               2 : twoModels,
               3: threeModels,
               4 : fourModels
    }
    
    options[numModels]()

    
    
if __name__ == '__main__':
   # plot(sys.argv[1:])
   plot1Models(['HyperDenseNet_2Mod'])