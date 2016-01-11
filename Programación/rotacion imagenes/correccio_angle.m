function y= correccio_angle(I,archivo)
 I= imread(I);
% 1.	Detectar els contorns en angles propers a la horitzontalitat. 
 I = rgb2gray(I);
 
%  Create a binary image. contorns
 BW = edge(I,'canny'); 
% Create the Hough transform using the binary image.
ang_inici=-5;
ing_fin=5;
[H,T,R] = hough(BW,'Theta',ang_inici:0.5:ing_fin );
%Trobar els pics
P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
x=T(P(:,2));%angles on es troben els pics del resultat de la transformada.

%Omplir contador de dos dimensions (angles, vots)
contador(1,:)=x(:);
contador(2,:)=0;
 
%Contar els vots de cada pic
 for i=1:size(x,2) 
     for j=1:size(x,2)   
         if contador(1,i)==x(j)
             contador(2,i)=contador(2,i)+1;
         end
     end
 end
 %Inicialitzem vot a 0
 vot=0;
 %Trobar quin angle té més vots
for k=1:size(contador,2)
   if contador(2,k) > vot
       vot=contador(2,k);
       ang_rot=contador(1,k);
   end
end
% 3.Rotar la imatge segons aquest angle.
% Rotate the image.
 rotI = imrotate(I,ang_rot,'crop');
% Mostra comparativa entre original i resultant
 imwrite(rotI,archivo)


end

