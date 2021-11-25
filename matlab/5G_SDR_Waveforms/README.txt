folder contains received data files for different transmitters
e.g. txPluto1_rxPluto0_txfile50_1.mat   used "Pluto1" as transmitter while "Pluto0" was used to receive. 
The waveform used to transmit is find in the .mat that starts with "50"
Each file contains the following variables:
"rxdata", "cntrFreq","sampleRate","datavalid","overflow","rx_gain","ts"
"rxdata" = this is the data that will be used to train/test the AI method
"cntrFreq"  - The data is all converted to baseband, but this is the parameter used for RF sampling with the SDR 
"sampleRate", Fs - clock rate used to sample the data at the receiver (also the upsampling rate of the transmitter)
"datavalid", 
"overflow",
"rx_gain", - not really used by the Pluto Receiver
"ts" - vector of time samples