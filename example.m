% Example for adaboost.m
clear all
close all
clc
% Type "edit adaboost.m" to see the code
max_hoods = 5;
 
   
 
% Make training data of two classes "red" and "blue"
 % with 2 features for each sample (the position  x and y).
  angle=rand(200,1)*2*pi; l=rand(200,1)*40+30; blue=[sin(angle).*l cos(angle).*l];
  angle=rand(200,1)*2*pi; l=rand(200,1)*40;    red=[sin(angle).*l cos(angle).*l];
 
 % All the training data
 load('BostonListing.mat')
 [C,ia] = unique(neighbourhood);
u_n = length(C);
C = string(C);
N = string(neighbourhood);
[s_ia,ind] = sort(ia);
for g=1:u_n
    n_c = strcmp(C(g),N);
    indx = find(n_c==1);
    class_assign = find(ind==g);
    truth(indx) = class_assign; % ind==g tells me where in neigbhourhood this hood is and assins class number.. farther down the list in actual == larger class num
end
 
truth_vec = truth;
% keeping only the "max hoods" number of  biggest neighborhoods
 
keep = [];
k = [];
for t_h = 1:max_hoods
    top_hoods(t_h) = mode(truth_vec);
    indx_mode = find(truth_vec==top_hoods(t_h));
    keep_truth_label = [keep, truth(indx_mode)];
    keep_x_training = [k,indx_mode];
    truth_vec(indx_mode) = [];
end
 
    X = [longitude,latitude];
    count=1;
    for ovo=1:length(top_hoods)
        for ovo_2 = 1:length(top_hoods)
            if ovo<ovo_2
              truth_1 = ones(length(find(truth==top_hoods(ovo))),1);
              train1_length = 0.8*length(truth_1);
              train1_length = floor(train1_length);
              truth_1_train = truth_1(1:train1_length); % 80% for training data 
              truth_1_test = truth_1(train1_length+1:end); % 20% for testing data 
 
              truth_2 = -1.*(ones(length(find(truth==top_hoods(ovo_2))),1));
              train2_length = 0.8*length(truth_2);
              train2_length = floor(train2_length);
              truth_2_train = truth_2(1:train2_length); % 80% for training data 
              truth_2_test = truth_2(train2_length+1:end); % 20% for testing data 
 
              one = X(find(truth==top_hoods(ovo)),:); % class one entire set
              one_train = one(1:train1_length,:);
              one_test = one(train1_length+1:end,:);
 
              two = X(find(truth==top_hoods(ovo_2)),:); % class two entire set
              two_train = two(1:train2_length,:);
              two_test = two(train2_length+1:end,:);
              
              OVO(count).train = [one_train;two_train]; % for OVO
              OVO(count).pair = [ovo,ovo_2];
              OVO_truth(count).train = [truth_1_train',truth_2_train'];
              
              OVO(count).test = [one_test;two_test];
              OVO_truth(count).test = [truth_1_test',truth_2_test'];
 
              count=count+1;
            end
            
        end
        
              truth_others = -1.*(ones(length(find(truth~=top_hoods(ovo))),1));
              train_others_length = 0.8*length(truth_others);
              train_others_length = floor(train_others_length);
              
              truth_others_train = truth_others(1:train_others_length); % 80% for training data 
              truth_others_test = truth_others(train_others_length+1:end); % 20% for testing data 
              
              others = X(find(truth~=top_hoods(ovo)),:); % class all others
              
              others_train = others(1:train_others_length,:); 
              others_test = others(train_others_length+1:end,:);
              
     OVA(ovo).train = [one_train;others_train]; % for OVA
     OVA(ovo).pair = [ovo,ovo_2];
     OVA(ovo).test = [one_test;others_test];
     OVA_truth(ovo).train = [truth_1_train',truth_others_train'];
     OVA_truth(ovo).test = [truth_1_test',truth_others_test'];
 
 
 
 
    end
num_runs = 1+max_hoods*2+max_hoods; % first run is normal example, rest are OVO or OVA.
for run = 1:num_runs
    
if run < 2 % toy problem
  datafeatures=[blue;red];
  truth_1_train = ones(length(blue),1);
  truth_2_train = -1.*ones(length(red),1);
  angle=rand(200,1)*2*pi; l=rand(200,1)*70; testdata=[sin(angle).*l cos(angle).*l];
  dataclass=[truth_1_train',truth_2_train'];
  testtruth=[ones(100,1);-1.*ones(100,1)];
  
elseif run > 1 && run < max_hoods*2+2 % OVO's
    datafeatures = [OVO(run-1).train];
    dataclass = [OVO_truth(run-1).train];
    testdata = [OVO(run-1).test];
    testtruth = [OVO_truth(run-1).test]';
else % OVA's
    datafeatures = [OVA(run-max_hoods*2-1).train];
    dataclass = [OVA_truth(run-max_hoods*2-1).train];
    testdata = [OVA(run-max_hoods*2-1).test];
    testtruth = [OVA_truth(run-max_hoods*2-1).test]';
end
 
 
 % Show the data
  figure, subplot(2,2,1), hold on; axis equal;
  blue=datafeatures(dataclass==-1,:); red=datafeatures(dataclass==1,:);
  plot(blue(:,1),blue(:,2),'b.'); plot(red(:,1),red(:,2),'r.');
  title(['Training Data for run', num2str(run)]);
  if run>1
      plot_google_map
  end
  
  
 % Use Adaboost to make a classifier
  [classestimate,model]=adaboost('train',datafeatures,dataclass,50);
  
  %svm OVA looking
  C=10;
  sigma=16;
  if run > 11
  SVMstruct = svmtrain(datafeatures,dataclass,'autoscale',false);
   SVMstruct_rbf = svmtrain(datafeatures,dataclass,'autoscale',false,'boxconstraint',C,'kernel_function','rbf','rbf_sigma',sigma); % train OVA btw class ovo and all else
  end
 
 % Training results
 % Show results
  blue=datafeatures(classestimate==-1,:); red=datafeatures(classestimate==1,:);
  
 
 % line practice
 % if dimension is 1, x so vertical line. if dim = 2, y, horizontal line
 
 h_line_x = [];
 h_line_y = [];
 v_line_y = [];
 v_line_x = [];
 
 subplot(2,2,2)
   for i=1:length(model)
      if(model(i).dimension==1) % x axis so vertical line
         h_line_x = [model(i).threshold,model(i).threshold];
         h_line_y = [min(datafeatures(:,2)),max(datafeatures(:,2))];
         plot(h_line_x,h_line_y)
         hold on
      else
          v_line_y = [model(i).threshold,model(i).threshold];
          v_line_x = [min(datafeatures(:,1)),max(datafeatures(:,1))];
          plot(v_line_x,v_line_y)
          hold on
      end
   end
          
 
 hold on
 plot(blue(:,1),blue(:,2),'b.');
 plot(red(:,1),red(:,2),'r.');
 title('weak classifiers - Ada boost');
 
         
 
 % Show the error verus number of weak classifiers
 error=zeros(1,length(model)); for i=1:length(model), error(i)=model(i).error; end 
 subplot(2,2,3), plot(error); title('Classification error versus number of weak classifiers');
 
 
 % Classify the testdata with the trained model
  testclass=adaboost('apply',testdata,model);
  
  % svm test
  if run > 11
 grouppredict = svmclassify(SVMstruct,testdata);
 grouppredict_rbf = svmclassify(SVMstruct_rbf,testdata);
  end

 
 % Show result
  blue=testdata(testclass==-1,:); red=testdata(testclass==1,:);
  
  % ccr
  
  ccr_1 = (testclass==testtruth);
  ccr_2 = find(ccr_1==1);
  ccr = length(ccr_2)/length(testdata);
  disp('////////////////////////////')
  disp(['ccr for run ',num2str(run)])
  ccr*100
  
  % ccr svm
  if run > 11
  ccr_svm_1 = (grouppredict==testtruth);
  ccr_svm_2 = (find(ccr_svm_1==1));
  ccr_svm = length(ccr_svm_2)/length(grouppredict);
  disp('svm ccr')
  ccr_svm*100
  
  ccr_svm_1_r = (grouppredict_rbf==testtruth);
  ccr_svm_2_r = (find(ccr_svm_1_r==1));
  ccr_svm_r = length(ccr_svm_2_r)/length(grouppredict_rbf);
  disp('svm ccr rbf')
  ccr_svm_r*100
  end
 
 % Show the data
  subplot(2,2,4), hold on
  plot(blue(:,1),blue(:,2),'b*');
  plot(red(:,1),red(:,2),'r*');
  axis equal;
  title('Test Data classified with adaboost model');
  if run>1
      plot_google_map
  end
  
 
 
end
