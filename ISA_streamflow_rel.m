% Wrapper for Instance Space Analysis for paper: Grey et al 2025 Journal of Hydrology,
% Harnessing the strengths of machine learning and geostatistics to improve streamflow prediction in ungauged basins; the best of both worlds
% 
% ISA Matlab code that the below wrapper uses can be accessed from:
% matilda.unimelb.edu.au
%
% Prediction of best performing algorithm (relative performance)

%% Section 1 = train model based on parameters of best performing model

% Relative
inputdir = './input_MM_PV/';
opts.rho = 0.1; %0.1
opts.k = 6; %7
opts.perf.AbsPerf = false;           % True if an absolute performance measure, False if a relative performance measure
opts.perf.epsilon = 0.00;           % Threshold of good performance
opts_rho_char = int2str(opts.rho * 10);
opts_k_char = int2str(opts.k);
rootdir = append(inputdir,'FINAL_MOD_rel_0/');
datafile_predict = [rootdir 'metadata_predict.csv'];


% code below here from loops in "ISA_flow_script.m"

%opts.rho = rho_list(i_rho);
%opts.k = k_list(j_k);


%opts_rho_char = int2str(opts.rho * 10);
%opts_k_char = int2str(opts.k);

% Create new rootdirectory, and save a copy of metadata.csv here
%readdatafile = readtable([inputdir 'metadata.csv']);

%rootdir = append(inputdir,'FINAL_MOD_rel_0_rho',opts_rho_char,'_k',opts_k_char,'/');
%rootdir = append(inputdir,'out_',char(opts.k),'/')
%mkdir(rootdir);

%writetable(readdatafile, append(rootdir,'metadata.csv'))

% rest of code

opts.parallel.flag = false;
opts.parallel.ncores = 4;

opts.perf.MaxPerf = false;          % True if Y is a performance measure to maximize, False if it is a cost measure to minimise.
%opts.perf.AbsPerf = false;           % True if an absolute performance measure, False if a relative performance measure
%opts.perf.epsilon = 0.00;           % Threshold of good performance
opts.perf.betaThreshold = 0.55;     % Beta-easy threshold
opts.auto.preproc = true;           % Automatic preprocessing on. Set to false if you don't want any preprocessing
opts.bound.flag = true;             % Bound the outliers. True if you want to bound the outliers, false if you don't
opts.norm.flag = true;              % Normalize/Standarize the data. True if you want to apply Box-Cox and Z transformations to stabilize the variance and scale N(0,1)

opts.selvars.smallscaleflag = false; % True if you want to do a small scale experiment with a percentage of the available instances
opts.selvars.smallscale = 0.50;     % Percentage of instances to be kept for a small scale experiment
% You can also provide a file with the indexes of the instances to be used.
% This should be a csvfile with a single column of integer numbers that
% should be lower than the number of instances
opts.selvars.fileidxflag = false;
opts.selvars.fileidx = '';
opts.selvars.densityflag = false;
opts.selvars.mindistance = 0.1;
opts.selvars.type = 'Ftr&Good';

opts.sifted.flag = false;            % Automatic feature selectio on. Set to false if you don't want any feature selection.
%opts.sifted.rho = 0.2;              % Minimum correlation value acceptable between performance and a feature. Between 0 and 1
%opts.sifted.K = 8;                 % Number of final features. Ideally less than 10.
opts.sifted.rho = opts.rho;
opts.sifted.K = opts.k;
opts.sifted.NTREES = 50;            % Number of trees for the Random Forest (to determine highest separability in the 2-d projection)
opts.sifted.MaxIter = 1000;
opts.sifted.Replicates = 100;

opts.pilot.analytic = false;        % Calculate the analytical or numerical solution
opts.pilot.ntries = 30;              % Number of attempts carried out by PBLDR

opts.cloister.pval = 0.05;
opts.cloister.cthres = 0.7;

opts.pythia.flag = true;
opts.pythia.useknn = false;
opts.pythia.cvfolds = 10;
opts.pythia.ispolykrnl = false;
opts.pythia.useweights = false;
opts.pythia.uselibsvm = true;

opts.trace.usesim = false;           % Use the actual or simulated data to calculate the footprints
opts.trace.PI = 0.50;               % Purity threshold, reduced from default 0.7

opts.outputs.csv = true;            %
opts.outputs.web = false;            % NOTE: MAKE THIS FALSE IF YOU ARE USING THIS CODE LOCALY - This flag is only useful if the system is being used 'online' through matilda.unimelb.edu.au
opts.outputs.png = false;            %

% create new saving directory
%save_dir = append('out_',opts.rho,'_',opts.k)
%save_dir = append('out_',string(opts.rho),'_',string(opts.k))
%save_dir = append(rootdir,'/out_test_',string(opts.k))
%mkdir(save_dir)
%rootdirSave = save_dir

% Saving all the information as a JSON file
fid = fopen([rootdir 'options.json'],'w+');
%fid = fopen(append(rootdir,'options.json'),'w+');
fprintf(fid,'%s',jsonencode(opts));
fclose(fid);

try
    model = buildIS(rootdir);
catch ME
    disp('EOF:ERROR');
    rethrow(ME)
end


% Section 2 = predict at sites across the catchment

Xbar = readtable(datafile_predict);

varlabels = Xbar.Properties.VariableNames;
isname = strcmpi(varlabels,'instances');
isfeat = strncmpi(varlabels,'feature_',8);
isalgo = strncmpi(varlabels,'algo_',5);
issource = strcmpi(varlabels,'source');
out.data.instlabels = Xbar{:,isname};
if isnumeric(out.data.instlabels)
    out.data.instlabels = num2cell(out.data.instlabels);
    out.data.instlabels = cellfun(@(x) num2str(x),out.data.instlabels,'UniformOutput',false);
end
if any(issource)
    out.data.S = categorical(Xbar{:,issource});
end
out.data.X = Xbar{:,isfeat};
out.data.Y = Xbar{:,isalgo};

out.data.Y = str2double(out.data.Y);

[ninst,nalgos] = size(out.data.Y);
% -------------------------------------------------------------------------
% HERE CHECK IF THE NUMBER OF ALGORITHMS IS THE SAME AS IN THE MODEL. IF
% NOT, CHECK IF THE NAMES OF THE ALGORITHMS ARE THE SAME, IF NOT, MOVE THE
% DATA IN SUCH WAY THAT THE NON-EXISTING ALGORITHMS ARE MADE NAN AND THE
% NEW ALGORITHMS ARE LAST.
out.data.algolabels = strrep(varlabels(isalgo),'algo_','');
algoexist = zeros(1,nalgos);
for ii=1:nalgos
    aux = find(strcmp(out.data.algolabels{ii},model.data.algolabels));
    if ~isempty(aux)
        algoexist(ii) = aux;
    end
end
newalgos = sum(algoexist==0);
modelalgos = length(model.data.algolabels);
Yaux = NaN+ones(ninst, modelalgos+newalgos);
lblaux = model.data.algolabels;
acc = modelalgos+1;
for ii=1:nalgos
    if algoexist(ii)==0
       Yaux(:,acc) = out.data.Y(:,ii);
       lblaux(:,acc) = out.data.algolabels(ii);
       acc = acc+1;
    else
        Yaux(:,algoexist(ii)) = out.data.Y(:,ii);
        % lblaux(:,acc) = out.data.algolabels(ii);
    end
end
out.data.Y = Yaux;
out.data.algolabels = lblaux;
nalgos = size(out.data.Y,2);
% -------------------------------------------------------------------------
% Storing the raw data for further processing, e.g., graphs
out.data.Xraw = out.data.X;
out.data.Yraw = out.data.Y;

% -------------------------------------------------------------------------
% Determine whether the performance of an algorithm is a cost measure to
% be minimized or a profit measure to be maximized. Moreover, determine
% whether we are using an absolute threshold as good peformance (the
% algorithm has a performance better than the threshold) or a relative
% performance (the algorithm has a performance that is similar that the
% best algorithm minus a percentage).
disp('-------------------------------------------------------------------------');
disp('-> Calculating the binary measure of performance');
msg = '-> An algorithm is good if its performace is ';
MaxPerf = false;
if isfield(model.opts.perf, 'MaxPerf')
    MaxPerf = model.opts.perf.MaxPerf;
elseif  isfield(model.opts.perf, 'MaxMin')
    MaxPerf = model.opts.perf.MaxMin;
else
    warning('Can not find parameter "MaxPerf" in the trained model. We are assuming that performance metric is needed to be minimized.');
end
if MaxPerf
    Yaux = out.data.Y;
    Yaux(isnan(Yaux)) = -Inf;
    [rankPerf,rankAlgo] = sort(Yaux,2,'descend');
    out.data.bestPerformace = rankPerf(:,1);
    out.data.P = rankAlgo(:,1);
    if model.opts.perf.AbsPerf
        out.data.Ybin = out.data.Y>=model.opts.perf.epsilon;
        msg = [msg 'higher than ' num2str(model.opts.perf.epsilon)];
    else
        out.data.bestPerformace(out.data.bestPerformace==0) = eps;
        out.data.Y(out.data.Y==0) = eps;
        out.data.Y = 1-bsxfun(@rdivide,out.data.Y,out.data.bestPerformace);
        out.data.Ybin = (1-bsxfun(@rdivide,Yaux,out.data.bestPerformace))<=model.opts.perf.epsilon;
        msg = [msg 'within ' num2str(round(100.*model.opts.perf.epsilon)) '% of the best.'];
    end
else
    Yaux = out.data.Y;
    Yaux(isnan(Yaux)) = Inf;
    [rankPerf,rankAlgo] = sort(Yaux,2,'ascend');
    out.data.bestPerformace = rankPerf(:,1);
    out.data.P = rankAlgo(:,1);
    if model.opts.perf.AbsPerf
        out.data.Ybin = out.data.Y<=model.opts.perf.epsilon;
        msg = [msg 'less than ' num2str(model.opts.perf.epsilon)];
    else
        out.data.bestPerformace(out.data.bestPerformace==0) = eps;
        out.data.Y(out.data.Y==0) = eps;
        out.data.Y = bsxfun(@rdivide,out.data.Y,out.data.bestPerformace)-1;
        out.data.Ybin = (bsxfun(@rdivide,Yaux,out.data.bestPerformace)-1)<=model.opts.perf.epsilon;
        msg = [msg 'within ' num2str(round(100.*model.opts.perf.epsilon)) '% of the best.'];
    end
end
disp(msg);
out.data.numGoodAlgos = sum(out.data.Ybin,2);
out.data.beta = out.data.numGoodAlgos>model.opts.perf.betaThreshold*nalgos;
% ---------------------------------------------------------------------
% Automated pre-processing
if model.opts.auto.preproc && model.opts.bound.flag
    disp('-------------------------------------------------------------------------');
    disp('-> Auto-pre-processing. Bounding outliers, scaling and normalizing the data.');
    % Eliminate extreme outliers, i.e., any point that exceedes 5 times the
    % inter quantile range, by bounding them to that value.
    disp('-> Removing extreme outliers from the feature values.');
    himask = bsxfun(@gt,out.data.X,model.prelim.hibound);
    lomask = bsxfun(@lt,out.data.X,model.prelim.lobound);
    out.data.X = out.data.X.*~(himask | lomask) + bsxfun(@times,himask,model.prelim.hibound) + ...
                                                  bsxfun(@times,lomask,model.prelim.lobound);
end

if model.opts.auto.preproc && model.opts.norm.flag
    % Normalize the data using Box-Cox and out.pilot.Z-transformations
    disp('-> Auto-normalizing the data.');
    out.data.X = bsxfun(@minus,out.data.X,model.prelim.minX)+1;

    for ii=1:length(model.prelim.lambdaX)
        %out.data.X(:,ii) = boxcox(out.data.X(:,ii),model.prelim.lambdaX(:,ii));
        out.data.X(:,ii) = boxcox(model.prelim.lambdaX(:,ii),out.data.X(:,ii));

    end
    out.data.X = bsxfun(@rdivide,bsxfun(@minus,out.data.X,model.prelim.muX),model.prelim.sigmaX);
    
    % If the algorithm is new, something else should be made...
   % out.data.Y(out.data.Y==0) = eps; % Assumes that out.data.Y is always positive and higher than 1e-16
   % for ii=1:modelalgos
   %     out.data.Y(:,ii) = boxcox(out.data.Y(:,ii),model.prelim.lambdaY);
   % end
   % out.data.Y(:,1:modelalgos) = bsxfun(@rdivide,bsxfun(@minus,out.data.Y(:,1:modelalgos),model.prelim.muY),model.prelim.sigmaY);
   % if newalgos>0
   %     [~,out.data.Y(:,modelalgos+1:nalgos),out.norm] = autoNormalize(ones(ninst,1), ... % Dummy variable
   %                                                                    out.data.Y(:,modelalgos+1:nalgos));
   % end
end
% ---------------------------------------------------------------------
% This is the final subset of features.
out.featsel.idx = model.featsel.idx;
out.data.X = out.data.X(:,out.featsel.idx);
out.data.featlabels = strrep(varlabels(isfeat),'feature_','');
out.data.featlabels = out.data.featlabels(model.featsel.idx);
% ---------------------------------------------------------------------
%  Calculate the two dimensional projection using the PBLDR algorithm
%  (Munoz et al. Mach Learn 2018)
out.pilot.Z = out.data.X*model.pilot.A';
% -------------------------------------------------------------------------
% Algorithm selection. Fit a model that would separate the space into
% classes of good and bad performance. 
out.pythia = PYTHIAtest(model.pythia, out.pilot.Z, out.data.Yraw, ...
                        out.data.Ybin, out.data.bestPerformace, ...
                        out.data.algolabels);
% -------------------------------------------------------------------------
% Validating the footprints
if model.opts.trace.usesim
    out.trace = TRACEtest(model.trace, out.pilot.Z, out.pythia.Yhat, ...
                          out.pythia.selection0, out.data.beta, ...
                          out.data.algolabels);
%     out.trace = TRACE(out.pilot.Z, out.pythia.Yhat, out.pythia.selection0, ...
%                       out.data.beta, out.data.algolabels, model.opts.trace);
else
    out.trace = TRACEtest(model.trace, out.pilot.Z, out.data.Ybin, ...
                          out.data.P, out.data.beta, ...
                          out.data.algolabels);
%     out.trace = TRACE(out.pilot.Z, out.data.Ybin, out.data.P, out.data.beta,...
%                       out.data.algolabels, model.opts.trace);
end

out.opts = model.opts;

% Saving results (my code)

predictCoords = out.data.instlabels;
predictCoords(:,2:3) = num2cell(out.pilot.Z(:,1:2));

writecell(predictCoords, ...
               [rootdir 'predict_coordinates.csv']);
 
predictYhat = out.data.instlabels;
predictYhat(:,2:3) = num2cell(out.pythia.Yhat(:,1:2));
predictYhat(:,4) = num2cell(out.pythia.selection0);

writecell(predictYhat, ...
               [rootdir 'predict_algorithm_svm.csv']);

disp("Finished")

% =========================================================================

