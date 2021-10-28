clear all
close all
num_Txrs = 15;
flag_debug = 1;
num_samples = 1e4;
%% input parameters for nonlinearities: phaseOffset, freqOffset, phNzFreqOff, ampGain, sampRate
% input parameter:   Range assumed reasonable (units)
% phaseOffset:       0-90; (degrees)
% freqOffset:        TBD maybe 1-100; must be>0(Hz) Note: freqOffset smears constellation around unit circle (constant modulus signals)
% phNzFreqOff(1)     TBD (maybe -35 to -60 <0 phNzFreqOff(1) [dBc/Hz] Phase noise level,  NOTE: more negative = less error introduced
% phNzFreqOff(2)     1-1000 [Hz] Frequency offset 
% Ia;                0-TBD 3; %[dB] Amplitude imbalance
% Ip;                0-TBD 3; %[degrees] Phase imbalance
% IDC;               0-TBD 0.01; %[?] DC offset applied to I
% QDC;               0-TBD -0.02; %[?] DC offset applied to IQ
% ampGain:           0-50 [dB]  0=do nothing, 50= noticable change in scatter plot


%% (digital) Transmit Signal Definition
% note: will replace this with 5G signal
M = 4;
data = randi([0 (M - 1)],num_samples,1);
pindBm = 10; % (dBm) signal power level
pin = 10.^((pindBm-30)/10); % Convert dBm to linear Watts
x = qammod(data,M,'UnitAveragePower',true)*sqrt(pin); %modulate data and set input power level (pin)
if(flag_debug)
    scatterplot(x); title('Input Signal')
end

%% define limits to parameters 
phaseOffset_limit = [0,30]; %[degrees]
freqOffset_limit = [1,100]; %50; %[Hz]
phNzFreqOff1_limit = [-1,-50];%-50;
phNzFreqOff2_limit = [1,1e3];
Ia_limit = [1, 10];%3; %[dB] Amplitude imbalance
Ip_limit = [1, 10];%3; %[degrees] Phase imbalance
IDC_limit = [0.0001, 0.1];%0.01; %[?] DC offset applied to I
QDC_limit = [-0.0001, -0.1];%-0.02; %[?] DC offset applied to IQ
ampGain_vec = [0, 50]; %20;

phaseOffset = [];
freqOffset  = []; 
phNzFreqOff1 = []; 
phNzFreqOff2 = []; 
Ia          = [];
Ip          = [];
IDC         = [];
QDC         = [];
ampGain     = [];
X_nl        = {};
%%  randomize units within limits set above
for i = 1:num_Txrs
    %% find parameters for nonlinearities from rand number generator
    %  x_out = randRange([xmin, xmax])
    phaseOffset = [phaseOffset randRange(phaseOffset_limit)];
    freqOffset = [freqOffset randRange(freqOffset_limit)];
    phNzFreqOff1 = [phNzFreqOff1 randRange(phNzFreqOff1_limit)];
    phNzFreqOff2 = [phNzFreqOff2 randRange(phNzFreqOff2_limit)];
    Ia = [Ia randRange(Ia_limit)];
    Ip = [Ip randRange(Ip_limit)];
    IDC = [IDC randRange(IDC_limit)];
    QDC = [QDC randRange(QDC_limit)];
    ampGain = [ampGain randRange(ampGain_vec)];
    
    IQ_imbalance = [Ia(i), Ip(i), IDC(i), QDC(i)];
    phNzFreqOff_vec = [phNzFreqOff1(i) phNzFreqOff2(i)];
    
    %% add nonlinearities to signal
    flag_debug=1;
    [x_nl] =addNonlinearities(x, phaseOffset(i), freqOffset(i), phNzFreqOff_vec, IQ_imbalance, ampGain(i), flag_debug);
    
    %% add awgn ??? (TBD)
    
    %% save off each signal appropriately
    X_nl{i} = x_nl;
    x=x_nl
    filename = ['tx',num2str(i),'.mat']
    tx_phaseOffset = phaseOffset(i);
    tx_freqOffset = freqOffset(i); 
    tx_phNzFreqOff_vec = phNzFreqOff_vec; 
    tx_IQ_imbalance = IQ_imbalance; 
    tx_ampGain = ampGain(i);
    
    save(filename,'x','tx_phaseOffset', 'tx_freqOffset','tx_phNzFreqOff_vec', 'tx_IQ_imbalance','tx_ampGain')
end





