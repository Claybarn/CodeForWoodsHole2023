addpath(genpath('C:\Users\Administrator\Desktop\CodeForWoodsHole2023'));

spike2Data = loadSpike2Data('C:\Users\Administrator\Desktop\data\07_05_test/test#1 purple_23-226_07+5.smrx','cedpath','C:\Users\Administrator\Desktop\CodeForWoodsHole2023\Meso\MesoProcessing-master\process_spike2\CEDS64ML');

pSpike2Data = processSpike2Data(spike2Data);

