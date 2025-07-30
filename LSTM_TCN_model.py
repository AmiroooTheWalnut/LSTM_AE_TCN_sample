import torch
import torch.nn as nn
from numpy import genfromtxt
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from pathlib import Path
import torch.optim as optim

class LSTM_TCN_fcn_1room:
    def run(self,suffix, inputFileName, genco2r1FileName, genoccr1FileName):
        # suffix = inputFileName[:-4]
        # Configuration
        modelVersion = 'LSTM_AE_TCN_1room'
        continueTrain = False
        isRunOnCPU = False
        batch_size = 1024
        seq_len = 21
        feature_dim = 1
        latent_dim = 256  # size of compressed vector

        # -----------------------------
        # Define the LSTM Autoencoder
        # -----------------------------
        class LSTMTCNAutoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim, seq_len, num_classes):
                super(LSTMTCNAutoencoder, self).__init__()
                self.seq_len = seq_len
                self.latent_dim = latent_dim

                # Encoder: returns final hidden + cell
                self.encoder_lstm = nn.LSTM(input_dim, latent_dim, batch_first=True)

                # Decoder input is the latent vector repeated seq_len times
                self.decoder_lstm = nn.LSTM(latent_dim, latent_dim, batch_first=True)
                self.directInput = nn.Linear(1, 64)
                self.activationInput = nn.Tanh()

                self.convSkip = nn.Conv1d(input_dim, 64, 1)

                self.conv1d1 = nn.Conv1d(input_dim, 64, 5, dilation=1, padding=2)
                self.conv1d2 = nn.Conv1d(64, 64, 5, dilation=2, padding=4)
                self.conv1d3 = nn.Conv1d(64, 64, 5, dilation=4, padding=8)
                self.conv1d4 = nn.Conv1d(64, 64, 5, dilation=8, padding=16)
                self.layerNorm1 = nn.LayerNorm(normalized_shape=[64, seq_len])
                self.layerNorm2 = nn.LayerNorm(normalized_shape=[64, seq_len])
                self.layerNorm3 = nn.LayerNorm(normalized_shape=[64, seq_len])
                self.layerNorm4 = nn.LayerNorm(normalized_shape=[64, seq_len])
                self.drop = nn.Dropout(p=0.01)
                self.activation = nn.ReLU()

                self.lastMaxPool = nn.MaxPool1d(5)
                self.flatten = nn.Flatten(start_dim=1, end_dim=2)

                self.output_layer = nn.Linear((latent_dim + 64)*2, num_classes)
                self.activationLast = nn.LeakyReLU()
                self.finalClassProb = nn.Softmax()

            def forward(self, x, xTCN, xDir):
                # Encode input
                _, (h_n, c_n) = self.encoder_lstm(x)  # h_n shape: (1, batch, latent_dim)

                # Repeat latent vector across time steps
                repeated_h = h_n.repeat(self.seq_len, 1, 1).permute(1, 0, 2)  # (batch, seq_len, latent_dim)

                # Decode repeated hidden state
                decoded_seq, (h_d, c_d) = self.decoder_lstm(repeated_h)
                outputHiddenState = torch.squeeze(h_d)
                dirX = self.directInput(xDir)
                dirXActivated = self.activationInput(dirX)
                mixedValsLSTM = torch.cat((outputHiddenState,dirXActivated),dim=1)

                convO = self.conv1d1(xTCN)
                norm = self.layerNorm1(convO)
                drop = self.drop(norm)
                act = self.activation(drop)

                skippedConv = self.convSkip(xTCN)
                addedPart1 = skippedConv + act

                convO = self.conv1d2(addedPart1)
                norm = self.layerNorm2(convO)
                drop = self.drop(norm)
                act = self.activation(drop)

                addedPart2 = addedPart1 + act

                convO = self.conv1d3(addedPart2)
                norm = self.layerNorm3(convO)
                drop = self.drop(norm)
                act = self.activation(drop)

                addedPart3 = addedPart2 + act

                convO = self.conv1d3(addedPart3)
                norm = self.layerNorm4(convO)
                drop = self.drop(norm)
                act = self.activation(drop)

                tcnBranch = self.lastMaxPool(act)
                flattenedTCNBranch = self.flatten(tcnBranch)

                dirX = self.directInput(xDir)
                dirXActivated = self.activationInput(dirX)
                mixedValsTCN = torch.cat((flattenedTCNBranch, dirXActivated), dim=1)

                mixedVals = torch.cat((mixedValsLSTM, mixedValsTCN), dim=1)

                out = self.output_layer(mixedVals)  # (batch, seq_len, input_dim)
                outActivated = self.activationLast(out)
                outClassified = self.finalClassProb(outActivated)
                return outClassified

        # % LOAD EXTRA GEN DATA
        r1co2Gen = genfromtxt(genco2r1FileName, delimiter=',',
                              skip_header=False)
        # r2co2Gen = genfromtxt(genco2r2FileName, delimiter=',',
        #                       skip_header=False)

        r1occGen = genfromtxt(genoccr1FileName, delimiter=',',
                              skip_header=False)
        # r2occGen = genfromtxt(genoccr2FileName, delimiter=',',
        #                       skip_header=False)

        counter = 0
        imgBatchesRawGen = numpy.zeros((r1co2Gen.shape[0] - seq_len, seq_len, feature_dim))
        rawDataCO21gen = []
        # rawDataCO22gen = []
        rawDataOcc1gen = []
        # rawDataOcc2gen = []
        for i in range((int)((seq_len - 1) / 2), r1co2Gen.shape[0] - ((int)((seq_len - 1) / 2)) - 1):
            for j in range(-((int)((seq_len - 1) / 2)), (int)((seq_len - 1) / 2)):
                imgBatchesRawGen[counter, j + (int)((seq_len - 1) / 2), 0] = r1co2Gen[i + j]
                # imgBatchesRawGen[counter, j + (int)((seq_len - 1) / 2), 1] = r2co2Gen[i + j]
            rawDataCO21gen.append(imgBatchesRawGen[counter, (int)((seq_len - 1) / 2), 0])
            # rawDataCO22gen.append(imgBatchesRawGen[counter, (int)((seq_len - 1) / 2), 1])
            rawDataOcc1gen.append(r1occGen[i])
            # rawDataOcc2gen.append(r2occGen[i])
            counter = counter + 1

        numInstances = counter

        genImgTrain = imgBatchesRawGen



        counter = 0
        imgBatchesRawGenTCN = numpy.zeros((r1co2Gen.shape[0] - seq_len, feature_dim, seq_len))
        # rawDataCO21gen = []
        # rawDataCO22gen = []
        # rawDataOcc1gen = []
        # rawDataOcc2gen = []
        for i in range((int)((seq_len - 1) / 2), r1co2Gen.shape[0] - ((int)((seq_len - 1) / 2)) - 1):
            for j in range(-((int)((seq_len - 1) / 2)), (int)((seq_len - 1) / 2)):
                imgBatchesRawGenTCN[counter, 0, j + (int)((seq_len - 1) / 2)] = r1co2Gen[i + j]
                # imgBatchesRawGenTCN[counter, 1, j + (int)((seq_len - 1) / 2)] = r2co2Gen[i + j]
            # rawDataCO21gen.append(imgBatchesRawGenTCN[counter, 0, (int)((seq_len - 1) / 2)])
            # rawDataCO22gen.append(imgBatchesRawGenTCN[counter, 1, (int)((seq_len - 1) / 2)])
            # rawDataOcc1gen.append(r1occGen[i])
            # rawDataOcc2gen.append(r2occGen[i])
            counter = counter + 1

        numInstances = counter

        genImgTrainTCN = imgBatchesRawGenTCN




        my_data = genfromtxt(inputFileName, delimiter=',', skip_header=True)

        counter = 0
        imgBatchesRawTest = numpy.zeros((my_data.shape[0] - seq_len, seq_len, feature_dim))  # imgWidth,2,1);
        rawDataCO21Test = []
        # rawDataCO22Test = []
        rawDataOcc1Test = []
        # rawDataOcc2Test = []
        for i in range((int)((seq_len - 1) / 2), my_data.shape[0] - ((int)((seq_len - 1) / 2)) - 1):
            for j in range(-((int)((seq_len - 1) / 2)), (int)((seq_len - 1) / 2)):
                imgBatchesRawTest[counter, j + (int)((seq_len - 1) / 2), 0] = my_data[i + j, 3]
                # imgBatchesRawTest[counter, j + (int)((seq_len - 1) / 2), 1] = my_data[i + j, 4]
            rawDataCO21Test.append(imgBatchesRawTest[counter, (int)((seq_len - 1) / 2), 0])
            # rawDataCO22Test.append(imgBatchesRawTest[counter, (int)((seq_len - 1) / 2), 1])
            rawDataOcc1Test.append(my_data[i, 0])
            # rawDataOcc2Test.append(my_data[i, 1])
            counter = counter + 1

        counter = 0
        imgBatchesRawTestTCN = numpy.zeros((my_data.shape[0] - seq_len, feature_dim, seq_len))  # imgWidth,2,1);
        # rawDataCO21Test = []
        # rawDataCO22Test = []
        # rawDataOcc1Test = []
        # rawDataOcc2Test = []
        for i in range((int)((seq_len - 1) / 2), my_data.shape[0] - ((int)((seq_len - 1) / 2)) - 1):
            for j in range(-((int)((seq_len - 1) / 2)), (int)((seq_len - 1) / 2)):
                imgBatchesRawTestTCN[counter, 0, j + (int)((seq_len - 1) / 2)] = my_data[i + j, 3]
                # imgBatchesRawTestTCN[counter, 1, j + (int)((seq_len - 1) / 2)] = my_data[i + j, 4]
            # rawDataCO21Test.append(imgBatchesRawTest[counter, (int)((seq_len - 1) / 2), 0])
            # rawDataCO22Test.append(imgBatchesRawTest[counter, (int)((seq_len - 1) / 2), 1])
            # rawDataOcc1Test.append(my_data[i, 0])
            # rawDataOcc2Test.append(my_data[i, 1])
            counter = counter + 1


        labelsTest = numpy.array(rawDataOcc1Test, dtype=numpy.int32)
        labelsTest[labelsTest > 2] = 2
        # torchLabels = torch.from_numpy(labels).to(torch.int64)
        # labelsOneHot = nn.functional.one_hot(torchLabels,3).to(torch.float32)
        # labelsOneHot = labelsOneHot.detach().cpu().numpy()
        encoded_arrTest = numpy.zeros((labelsTest.size, labelsTest.max() + 1), dtype=int)
        encoded_arrTest[numpy.arange(labelsTest.size), labelsTest] = 1
        labelsOneHotTest = encoded_arrTest
        xTest = torch.from_numpy(imgBatchesRawTest).to(dtype=torch.float32)
        xTestTCN = torch.from_numpy(imgBatchesRawTestTCN).to(dtype=torch.float32)

        # Create dummy input
        x = torch.from_numpy(genImgTrain).to(dtype=torch.float32)
        # x = torch.randn(batch_size, seq_len, feature_dim)  # (batch, seq_len, input_dim)

        # Create model
        model = LSTMTCNAutoencoder(input_dim=feature_dim, latent_dim=latent_dim, seq_len=seq_len, num_classes=3)

        labels = numpy.array(rawDataOcc1gen, dtype=numpy.int32)
        labels[labels > 2] = 2
        # torchLabels = torch.from_numpy(labels).to(torch.int64)
        # labelsOneHot = nn.functional.one_hot(torchLabels,3).to(torch.float32)
        # labelsOneHot = labelsOneHot.detach().cpu().numpy()
        encoded_arr = numpy.zeros((labels.size, labels.max() + 1), dtype=int)
        encoded_arr[numpy.arange(labels.size), labels] = 1
        labelsOneHot = encoded_arr

        model = LSTMTCNAutoencoder(feature_dim, latent_dim, seq_len, 3)

        my_file = Path('model' + modelVersion + suffix + '.pytorch')
        if my_file.is_file() and continueTrain == True:
            try:
                model.load_state_dict(torch.load('model' + modelVersion + suffix + '.pytorch', weights_only=True))
            except:
                print("FAILED TO LOAD WEGHTS!")
        if isRunOnCPU == False:
            model = model.cuda()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Training loop (1 epoch example)
        # model.train()
        indices = numpy.arange(genImgTrain.shape[0])
        counter = 1
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.992)
        for epoch in range(8000):
            chosen_indices = torch.from_numpy(numpy.random.choice(indices, size=batch_size, replace=False))
            batchSampledX = torch.from_numpy(genImgTrain[chosen_indices, :, :]).to(dtype=torch.float32)
            batchSampledXTCN = torch.from_numpy(genImgTrainTCN[chosen_indices, :, :]).to(dtype=torch.float32)
            batchSampledDirX1 = torch.from_numpy(numpy.array(rawDataCO21gen)[chosen_indices]).to(dtype=torch.float32)
            # batchSampledDirX2 = torch.from_numpy(numpy.array(rawDataCO22gen)[chosen_indices]).to(dtype=torch.float32)
            batchSampledDirX = torch.unsqueeze(batchSampledDirX1,dim=1)
            batchSampledLabels = torch.from_numpy(labelsOneHot[chosen_indices, :]).to(dtype=torch.float32)
            # batchSampledX = x.gather(dim=0,index=chosen_indices)
            # batchSampledLabels = labelsOneHot.gather(dim=0,index=chosen_indices)
            optimizer.zero_grad()
            if isRunOnCPU == False:
                batchSampledX = batchSampledX.cuda()
                batchSampledXTCN = batchSampledXTCN.cuda()
                batchSampledLabels = batchSampledLabels.cuda()
                batchSampledDirX = batchSampledDirX.cuda()
            output = model(batchSampledX, batchSampledXTCN, batchSampledDirX)
            loss = criterion(output, batchSampledLabels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if counter > 99:
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, lr: {scheduler.get_lr()}")
                counter = 0
            counter = counter + 1

        predLabels = numpy.argmax(output.detach().cpu().numpy(), axis=1)
        labels = numpy.argmax(batchSampledLabels.detach().cpu().numpy(), axis=1)
        cm = confusion_matrix(labels, predLabels)
        print("Train:")
        print(cm)
        accuracy = accuracy_score(labels, predLabels)
        print(accuracy)

        print("Test:")
        dirX1 = torch.from_numpy(numpy.array(rawDataCO21Test)).to(dtype=torch.float32)
        # dirX2 = torch.from_numpy(numpy.array(rawDataCO22Test)).to(dtype=torch.float32)
        dirX = torch.unsqueeze(dirX1,dim=1)
        if isRunOnCPU == False:
            xTest = xTest.cuda()
            xTestTCN = xTestTCN.cuda()
            dirX = dirX.cuda()
        outputTest = model(xTest,xTestTCN,dirX)

        predLabelsTest = numpy.argmax(outputTest.detach().cpu().numpy(), axis=1)
        # labelsTest = numpy.argmax(batchSampledLabels.detach().cpu().numpy(),axis=1)
        cmTest = confusion_matrix(labelsTest, predLabelsTest)
        print(cmTest)
        accuracyTest = accuracy_score(labelsTest, predLabelsTest)
        print(accuracyTest)

        torch.save(model.state_dict(), 'model' + modelVersion + suffix + '.pytorch')

        return accuracyTest
        # plt.plot(predLabels, label='train pred labels')
        # plt.plot(labels, label='train labels')
        # plt.legend()
        # plt.show()
