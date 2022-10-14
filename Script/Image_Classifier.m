function [x] = Image_Classifier()
%%---------------------------------------------------------PART (1) --------------------------------------------------------------------------%%


%lena
m1=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\01.jpeg');
xm= uint8(m1);
x=rgb2gray(xm);
A1 = imresize(x, [92, 112]);
%[nx1,my1]=size(A1);
At1 = fft2(A1);% convert image from SD to FD 
f1 = log(abs(fftshift(At1))+1);%Enhancment in the frequncy domain using log Transformation. why ? the explantion in the report 
F1=mat2gray(f1); % Using matgray to scale image between 0 and 1 
figure(1),subplot(2,2,1),imshow(F1,[]),title('Class A','fontSize',16); % displaying the FFt coefficients
 figure(2),subplot(2,2,1),surf(double(F1)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 

m2=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\02.jpg');
ym= uint8(m2);
y=rgb2gray(ym);
A2 = imresize(y, [92, 112]);
%[nx2,my2]=size(A2);
At2 = fft2(A2);% convert image from SD to FD 
f2 = log(abs(fftshift(At2))+1);%Enhancment in the frequncy domain using log Transformation.
F2=mat2gray(f2); % Using matgray to scale image between 0 and 1  
figure(1),subplot(2,2,2),imshow(F2,[]),title('Class A','fontSize',16); % displaying the FFt coefficients
 figure(2),subplot(2,2,2),surf(double(F2)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 
 
m3=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\03.jpg');
z = uint8(m3);
%z=rgb2gray(zm);
A3 = imresize(z, [92, 112]);
%[nx3,my3]=size(A3);
At3 = fft2(A3);% convert image from SD to FD 
f3 = log(abs(fftshift(At3))+1);%Enhancment in the frequncy domain using log Transformation.
F3=mat2gray(f3); % Using matgray to scale image between 0 and 1  
figure(1),subplot(2,2,3),imshow(F3,[]),title('Class A','fontSize',16); % displaying the FFt coefficients
 figure(2),subplot(2,2,3),surf(double(F3)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 
% A Class 
justforshow=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Testing set\11.png');
r=rgb2gray(justforshow);
figure(1),subplot(2,2,4),imshow(r),title('Test','fontSize',16);
 figure(2),subplot(2,2,4),surf(double(r)),colormap('gray');title('Histo of orginal','fontSize',16);

%% ----------------------------------------------------------------------------------------------------------------------------------
m4=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\11.jpg');
A4 = uint8(m4);
%[nx4,my4]=size(A4);
At4 = fft2(A4);% convert image from SD to FD 
f4 = log(abs(fftshift(At4))+1);%Enhancment in the frequncy domain using log Transformation.
F4=mat2gray(f4); % Using matgray to scale image between 0 and 1  
figure(3),subplot(2,2,1),imshow(F4,[]),title('Class B','fontSize',16); % displaying the FFt coefficients
 figure(4),subplot(2,2,1),surf(double(F4)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain


m5=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\12.jpg');
A5 = uint8(m5);
% [nx5,my5]=size(A5);
At5 = fft2(A5);% convert image from SD to FD 
f5 = log(abs(fftshift(At5))+1);%Enhancment in the frequncy domain using log Transformation.
F5=mat2gray(f5); % Using matgray to scale image between 0 and 1  
figure(3),subplot(2,2,2),imshow(F5,[]),title('Class B','fontSize',16); % displaying the FFt coefficients
 figure(4),subplot(2,2,2),surf(double(F5)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 
 
m6=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\13.jpg');
A6 = uint8(m6);
%[nx6,my6]=size(A6);
At6 = fft2(A6);% convert image from SD to FD 
f6 = log(abs(fftshift(At6))+1);%Enhancment in the frequncy domain using log Transformation.
F6=mat2gray(f6); % Using matgray to scale image between 0 and 1  
figure(3),subplot(2,2,3),imshow(F6,[]),title('Class B','fontSize',16); % displaying the FFt coefficients
 figure(4),subplot(2,2,3),surf(double(F6)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 
 %B Class 
 figure(3),subplot(2,2,4),imshow(A4),title('Person-2','fontSize',16);
  figure(4),subplot(2,2,4),surf(double(A4)),colormap('gray');title('Histo of orginal','fontSize',16);
 
 
%% ----------------------------------------------------------------------------------------------------------------------------------
m7=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\21.jpg');
A7 = uint8(m7);
%[nx7,my3]=size(A7);
At7 = fft2(A7);% convert image from SD to FD 
f7 = log(abs(fftshift(At7))+1);%Enhancment in the frequncy domain using log Transformation.
F7=mat2gray(f7); % Using matgray to scale image between 0 and 1  
figure(5),subplot(2,2,1),imshow(F7,[]),title('Class C','fontSize',16); % displaying the FFt coefficients
 figure(6),subplot(2,2,1),surf(double(F7)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 

m8=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\22.jpg');
A8= uint8(m8);
%[nx8,my8]=size(A8);
At8 = fft2(A8);% convert image from SD to FD 
f8 = log(abs(fftshift(At8))+1);%Enhancment in the frequncy domain using log Transformation.
F8=mat2gray(f8); % Using matgray to scale image between 0 and 1  
figure(5),subplot(2,2,2),imshow(F8,[]),title('Class C','fontSize',16); % displaying the FFt coefficients
 figure(6),subplot(2,2,2),surf(double(F8)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 


m9=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\23.jpg');
A9 = uint8(m9);
%[nx9,my9]=size(A9);
At9 = fft2(A9);% convert image from SD to FD 
f9 = log(abs(fftshift(At9))+1);%Enhancment in the frequncy domain using log Transformation.
F9=mat2gray(f9); % Using matgray to scale image between 0 and 1  
figure(5),subplot(2,2,3),imshow(F9,[]),title('Class C','fontSize',16); % displaying the FFt coefficients
 figure(6),subplot(2,2,3),surf(double(F9)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 
  %C Class 
 figure(5),subplot(2,2,4),imshow(A7),title('Person-3','fontSize',16);
  figure(6),subplot(2,2,4),surf(double(A7)),colormap('gray');title('Histo of orginal','fontSize',16);
 
 
 

%% ----------------------------------------------------------------------------------------------------------------------------------
m10=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\31.jpg');
A10 = uint8(m10);
%[nx10,my10]=size(A10);
At10 = fft2(A10);% convert image from SD to FD 
f10 = log(abs(fftshift(At10))+1);%Enhancment in the frequncy domain using log Transformation.
F10=mat2gray(f10); % Using matgray to scale image between 0 and 1  
figure(7),subplot(2,2,1),imshow(F10,[]),title('Class D','fontSize',16); % displaying the FFt coefficients
 figure(8),subplot(2,2,1),surf(double(F10)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 
 
m11=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\32.jpg');
A11 = uint8(m11);
%[nx11,my11]=size(A11);
At11 = fft2(A11);% convert image from SD to FD 
f11 = log(abs(fftshift(At11))+1);%Enhancment in the frequncy domain using log Transformation.
F11=mat2gray(f11); % Using matgray to scale image between 0 and 1  
figure(7),subplot(2,2,2),imshow(F11,[]),title('Class D','fontSize',16); % displaying the FFt coefficients
 figure(8),subplot(2,2,2),surf(double(F11)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain

m12=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\33.jpg');
A12 = uint8(m12);
%[nx12,my12]=size(A12);
At12 = fft2(A12);% convert image from SD to FD 
f12 = log(abs(fftshift(At12))+1);%Enhancment in the frequncy domain using log Transformation.
F12 =mat2gray(f12); % Using matgray to scale image between 0 and 1  
figure(7),subplot(2,2,3),imshow(F12,[]),title('Class D','fontSize',16); % displaying the FFt coefficients
 figure(8),subplot(2,2,3),surf(double(F12)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 
  
  %D Class 
 figure(7),subplot(2,2,4),imshow(A10),title('Person-4','fontSize',16);
  figure(8),subplot(2,2,4),surf(double(A10)),colormap('gray');title('Histo of orginal','fontSize',16);
 

%% ----------------------------------------------------------------------------------------------------------------------------------
m13=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\41.jpg');
A13 = uint8(m13);
%[nx13,my13]=size(A13);
At13 = fft2(A13);% convert image from SD to FD 
f13 = log(abs(fftshift(At13))+1);%Enhancment in the frequncy domain using log Transformation. 
F13=mat2gray(f13); % Using matgray to scale image between 0 and 1  
figure(9),subplot(2,2,1),imshow(F13,[]),title('Class E','fontSize',16); % displaying the FFt coefficients
 figure(10),subplot(2,2,1),surf(double(F13)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 

m14=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\42.jpg');
A14 = uint8(m14);
%[nx14,my14]=size(A14);
At14 = fft2(A14);% convert image from SD to FD 
f14 = log(abs(fftshift(At14))+1);%Enhancment in the frequncy domain using log Transformation.
F14=mat2gray(f14); % Using matgray to scale image between 0 and 1  
figure(9),subplot(2,2,2),imshow(F14,[]),title('Class E','fontSize',16); % displaying the FFt coefficients
 figure(10),subplot(2,2,2),surf(double(F14)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain
 
m15=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Training set\43.jpg');
A15 = uint8(m15);
%[nx15,my15]=size(A15);
At15 = fft2(A15);% convert image from SD to FD 
f15 = log(abs(fftshift(At15))+1);%Enhancment in the frequncy domain using log Transformation.
F15=mat2gray(f15); % Using matgray to scale image between 0 and 1  
figure(9),subplot(2,2,3),imshow(F15,[]),title('Class E','fontSize',16); % displaying the FFt coefficients
 figure(10),subplot(2,2,3),surf(double(F15)),colormap('gray');title('log in FD','fontSize',16);%displaying histogram of the image in frequncy domain

 %E Class 
 figure(9),subplot(2,2,4),imshow(A13),title('Person-5','fontSize',16);
  figure(10),subplot(2,2,4),surf(double(A13)),colormap('gray');title('Histo of orginal','fontSize',16);
 
%% ----------------------------------------------------------------------------------------------------------------------------------
%%---------------------------------------------------------PART (2) --------------------------------------------------------------------------%%
% --------------->> Training set Sorting and extaract features
%%------------->>Step(2) Sorting (Descend)  Image in FD then geting first 5 elements of matrix as feature vector <<--------------------------%%
f1sorted=sort(f1(:),'descend');
Tn1features=f1sorted(1:5,1);
f2sorted=sort(f2(:),'descend');
Tn2features=f2sorted(1:5,1);
f3sorted=sort(f3(:),'descend');
Tn3features=f3sorted(1:5,1);
f4sorted=sort(f4(:),'descend');
Tn4features=f4sorted(1:5,1);
f5sorted=sort(f5(:),'descend');
Tn5features=f5sorted(1:5,1);
f6sorted=sort(f6(:),'descend');
Tn6features=f6sorted(1:5,1);
f7sorted=sort(f7(:),'descend');
Tn7features=f7sorted(1:5,1);
f8sorted=sort(f8(:),'descend');
Tn8features=f8sorted(1:5,1);
f9sorted=sort(f9(:),'descend');
Tn9features=f9sorted(1:5,1);
f10sorted=sort(f10(:),'descend');
Tn10features=f10sorted(1:5,1);
f11sorted=sort(f11(:),'descend');
Tn11features=f11sorted(1:5,1);
f12sorted=sort(f12(:),'descend');
Tn12features=f12sorted(1:5,1);
f13sorted=sort(f13(:),'descend');
Tn13features=f13sorted(1:5,1);
f14sorted=sort(f14(:),'descend');
Tn14features=f14sorted(1:5,1);
f15sorted=sort(f15(:),'descend');
Tn15features=f15sorted(1:5,1);
%144301

%% ----------------------------------------------------------------------------------------------------------------------------------

% ---------------> Testing Image 
mT=imread('C:\Users\Muaad\Desktop\Muaad-Image-Assignment\DataSet\Testing set\11.png');
mt=uint8(mT);
%mt=rgb2gray(mtn);


% mtn = imnoise(mtnn,'salt & pepper',1);% Removing noise 
% mt= medfilt2(mtn);
att1=fft2(mt);% convert image to Frequncy domain 
ft1 = log(abs(fftshift(att1))+1); %Enhancment in the frequncy domain using log Transformation.
%Zero out all small ceofficints and inverse transfom 
% disp('Zeroing out small Fourier Coefficients')
% for thresh =0.1*[0.001 0.005 0.01]*max(max(abs(att1)));
%     ind = abs(att1)>thresh;
%     count = nx*ny - sum(sum(ind));
%     atlow=att1t*ind;
%     percent=100-count/(nx*ny)*100;
%     Alow=uint8(ifft2(atlow));
mtSorted=sort(ft1(:),'descend');% after getting ft1 sort the matrix to get first 5 element 
MTfeatures=mtSorted(1:5,1);% The large 5 value 
MTfeaturesT=MTfeatures'; % just transpose to be row wise 
%     title([num2str(percent) '% of FFT basis'],'FontSize',18)

%% ----------------------------------------------------------------------------------------------------------------------------------
% Transpose the features from column matrix  to row matrix to apply KNN  
Tn1featuresT=Tn1features';
  Tn2featuresT=Tn2features';
  Tn3featuresT=Tn3features';
  Tn4featuresT=Tn4features';
  Tn5featuresT=Tn5features';
  Tn6featuresT=Tn6features';
  Tn7featuresT=Tn7features';
  Tn8featuresT=Tn8features';
  Tn9featuresT=Tn9features';
  Tn10featuresT=Tn10features';
  Tn11featuresT=Tn11features';
  Tn12featuresT=Tn12features';
  Tn13featuresT=Tn13features';
  Tn14featuresT=Tn14features';
  Tn15featuresT=Tn15features';  
  
%% ----------------------------------------------------------------------------------------------------------------------------------
  % 5 vector of testing image 
 Tf1=MTfeaturesT(1,1);
 Tf2=MTfeaturesT(1,2);
 Tf3=MTfeaturesT(1,3);
 Tf4=MTfeaturesT(1,4);
 Tf5=MTfeaturesT(1,5);
 %% ----------------------------------------------------------------------------------------------------------------------------------
  disp('---------------------------------------------------------------------------------------------------------------------------------------------------');
  
 disp('--------------------------------->>Muaad144301_Image_Assignment_2<<----------------------------------------------------');

 %% ----------------------------------------------------------------------------------------------------------------------------------
 %%---------------------------------------------------------PART (3) --------------------------------------------------------------------------%%
    % to Classify we can use many classification techniques,
    %im used  k-nearest neighbors algorithm 
        %KNN Calculation of the euclidean distance 
        dis=zeros(15);
        dis(1)=sqrt( (Tf1-Tn1featuresT(1,1))^2  + (Tf2-Tn1featuresT(1,2))^2   + (Tf3-Tn1featuresT(1,3))^2   + (Tf4-Tn1featuresT(1,4))^2 + (Tf5-Tn1featuresT(1,5))^2 ); %Class A
        dis(2)=sqrt( (Tf1-Tn2featuresT(1,1))^2  + (Tf2-Tn2featuresT(1,2))^2   + (Tf3-Tn2featuresT(1,3))^2   + (Tf4-Tn2featuresT(1,4))^2 + (Tf5-Tn2featuresT(1,5))^2 ); %Class A
        dis(3)=sqrt( (Tf1-Tn3featuresT(1,1))^2  + (Tf2-Tn3featuresT(1,2))^2   + (Tf3-Tn3featuresT(1,3))^2   + (Tf4-Tn3featuresT(1,4))^2 + (Tf5-Tn3featuresT(1,5))^2 ); %Class A
        dis(4)=sqrt( (Tf1-Tn4featuresT(1,1))^2  + (Tf2-Tn4featuresT(1,2))^2   + (Tf3-Tn4featuresT(1,3))^2   + (Tf4-Tn4featuresT(1,4))^2 + (Tf5-Tn4featuresT(1,5))^2 ); %Class B
        dis(5)=sqrt( (Tf1-Tn5featuresT(1,1))^2  + (Tf2-Tn5featuresT(1,2))^2   + (Tf3-Tn5featuresT(1,3))^2   + (Tf4-Tn5featuresT(1,4))^2 + (Tf5-Tn5featuresT(1,5))^2 ); %Class B
        dis(6)=sqrt( (Tf1-Tn6featuresT(1,1))^2  + (Tf2-Tn6featuresT(1,2))^2   + (Tf3-Tn6featuresT(1,3))^2   + (Tf4-Tn6featuresT(1,4))^2 + (Tf5-Tn6featuresT(1,5))^2 ); %Class B
        dis(7)=sqrt( (Tf1-Tn7featuresT(1,1))^2  + (Tf2-Tn7featuresT(1,2))^2   + (Tf3-Tn7featuresT(1,3))^2   + (Tf4-Tn7featuresT(1,4))^2 + (Tf5-Tn7featuresT(1,5))^2 ); %Class C
        dis(8)=sqrt( (Tf1-Tn8featuresT(1,1))^2  + (Tf2-Tn8featuresT(1,2))^2   + (Tf3-Tn8featuresT(1,3))^2   + (Tf4-Tn8featuresT(1,4))^2 + (Tf5-Tn8featuresT(1,5))^2 ); %Class C
        dis(9)=sqrt( (Tf1-Tn9featuresT(1,1))^2  + (Tf2-Tn9featuresT(1,2))^2   + (Tf3-Tn9featuresT(1,3))^2   + (Tf4-Tn9featuresT(1,4))^2 + (Tf5-Tn9featuresT(1,5))^2 ); %Class C
        dis(10)=sqrt( (Tf1-Tn10featuresT(1,1))^2  + (Tf2-Tn10featuresT(1,2))^2   + (Tf3-Tn10featuresT(1,3))^2   + (Tf4-Tn10featuresT(1,4))^2 + (Tf5-Tn10featuresT(1,5))^2 ); %Class D
        dis(11)=sqrt( (Tf1-Tn11featuresT(1,1))^2  + (Tf2-Tn11featuresT(1,2))^2   + (Tf3-Tn11featuresT(1,3))^2   + (Tf4-Tn11featuresT(1,4))^2 + (Tf5-Tn11featuresT(1,5))^2 ); %Class D 
        dis(12)=sqrt( (Tf1-Tn12featuresT(1,1))^2  + (Tf2-Tn12featuresT(1,2))^2   + (Tf3-Tn12featuresT(1,3))^2   + (Tf4-Tn12featuresT(1,4))^2 + (Tf5-Tn12featuresT(1,5))^2 ); %Class D
        dis(13)=sqrt( (Tf1-Tn13featuresT(1,1))^2  + (Tf2-Tn13featuresT(1,2))^2   + (Tf3-Tn13featuresT(1,3))^2   + (Tf4-Tn13featuresT(1,4))^2 + (Tf5-Tn13featuresT(1,5))^2 ); %Class E
        dis(14)=sqrt( (Tf1-Tn14featuresT(1,1))^2  + (Tf2-Tn14featuresT(1,2))^2   + (Tf3-Tn14featuresT(1,3))^2   + (Tf4-Tn14featuresT(1,4))^2 + (Tf5-Tn14featuresT(1,5))^2 ); %Class E 
        dis(15)=sqrt( (Tf1-Tn15featuresT(1,1))^2  + (Tf2-Tn15featuresT(1,2))^2   + (Tf3-Tn15featuresT(1,3))^2   + (Tf4-Tn15featuresT(1,4))^2 + (Tf5-Tn15featuresT(1,5))^2 ); %Class E 
   
        %Dispalying the distances of all Classes in row wise 
        disp('---------------------------------------------------------------------------------------------------------------------------------------------------');
        disp('--------------------------->>Eclidaince Distances of all image in row wise<<--------------------------------------------------');
        disp('---------------------------------------------------------------------------------------------------------------------------------------------------');
        ClassA=[dis(1) dis(2) dis(3)];
        ClassB=[dis(4) dis(5) dis(6)];
        ClassC=[dis(7) dis(8) dis(9)];
        ClassD=[dis(10) dis(11) dis(12)];
        ClassE=[dis(13) dis(14) dis(15)];    
        
            disp('Class A --> '),disp(ClassA(1,:));
            disp('Class B --> '),disp(ClassB(1,:));
            disp('Class C --> '),disp(ClassC(1,:));
            disp('Class D --> '),disp(ClassD(1,:));
            disp('Class E --> '),disp(ClassE(1,:));
            
   %% ----------------------------------------------------------------------------------------------------------------------------------
         %Geting the min distances of 15 image 
        Class_Result=[1 ; 1; 1; 2; 2; 2; 3; 3; 3; 4; 4; 4; 5; 5; 5;];
        INF=99999;
        Answer=' ';
        for i=1:1:15
            if dis(i) <INF
                INF= dis(i); % taking the min by looping in all distance of 15 images
                Answer=Class_Result(i); % puting the testing image with the same class of the minimum distance. 
            end
        end 
        % just handling the string output !
        FinalResult=' ';
        if Answer==1
            FinalResult='Class A : Lena';
        end
        
        if Answer==2
            FinalResult='Class B : ';
        end
        
        if Answer==3
            FinalResult='Class C';
        end
        
        if Answer==4
            FinalResult='Class D';
        end
        
        if Answer==5
            FinalResult='Class E';
        end
        
     %% ----------------------------------------------------------------------------------------------------------------------------------
        %displaying result !
  disp('---------------------------------------------------------------------------------------------------------------------------------------------------');
disp( ['The Testing image will be in -----------> '  FinalResult]);
disp('---------------------------------------------------------------------------------------------------------------------------------------------------');


% END of Function ->
% 144301
end