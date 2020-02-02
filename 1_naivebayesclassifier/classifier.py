'''
Reusing the same NBC formulas from my thesis
since the dataset is binary. 
'''
import numpy as np
import cv2
import csv
import sys


#Load the images
im1 = np.load('class_a.npy')
im2 = np.load('class_b.npy')
im3 = np.load('field.npy')

'''
Using shape on each numpy file :
Assumption is class_a/_b has 1000 sets of 40x60 pixel following numpy array format (1000,40,60)
field has 200 sets of 40x60 pixel format (200,40,60)
#a = im3.shape 
print (np.size(im1,0))
print (np.size(im1,1))
print (np.size(im1,2))
'''



def nbc_train_classifier(test_line):
    #########################
    ##########Trainer########
    #########################
    #sum up all the '1' in both datasets,
    a_onesum=0
    b_onesum=0
    
    #reshape 40x60 into 2400 feature size
    new_im1 = im1.reshape(1000,2400)
    new_im2 = im2.reshape(1000,2400)

        
    #sum all 1's for each class
    for i in range(0, 999):
        for j in range(0, 2399):
                a_onesum += new_im1[i,j]
                b_onesum += new_im2[i,j]
    

    '''
    Maximum Likelihood Class Priors
    ML class priors pA/B
    Take number dataset for classA/Total dataset
    '''
    pA = 1000/2000
    pB = 1 - pA
    
    '''
    ML estimates 
    sum up all 1's for each class a/b 
    and divide with total dataset*feature size =1000x2400 for class a/b respectively
    '''
    phiA = ( a_onesum ) / ( 1000*2400)
    phiB = ( b_onesum ) / ( 1000*2400)
    minusphiA = 1-phiA
    minusphiB = 1-phiB
    
    #print (phiA)
    #print (phiB)
    
    #########################
    #######Classifier########
    #########################

    #Reshape the test dataset (field.npy)
    new_im3 = im3.reshape(200,2400) #
    
    #Need to create 2 version of test vector,test_vec and 1-test_vec
    test_vec= np.empty(shape=2400)
    one_minus_test_vec = np.empty(shape=2400)
     
    for i in range(0, 2399):
        test_vec[i]=new_im3[test_line,i]  #feed in 1 image at a time

    for i in range(0, 2399):
        one_minus_test_vec[i] = 1-test_vec[i]
        
    #Bernoulli modelling,calculate pxa and pxb
    pxa = (np.prod(np.float_power(phiA, test_vec)))*(np.prod(np.float_power(minusphiA,one_minus_test_vec)))
    pxb = (np.prod(np.float_power(phiB, test_vec)))*(np.prod(np.float_power(minusphiB,one_minus_test_vec)))
    
    #print(pxa)
    #print(pxb)
    
    #Class Probability Determination
    btm= (pxa*pA + pxb*pB)
   
    #If formula above correct, Prob_A and Prob_B will add to 1 
    #and shoudn't be extremely skewed to one class for this dataset
    Prob_A = pxa*pA / btm
    Prob_B = pxb*pB / btm
    
    
        
    #########################
    #######Evaluator#########
    #########################
    #Return predicted class of test_vector/image
    if (Prob_A > Prob_B) : 
        out = "A"
    else:
        out = "B"
            
    #store results
    results = np.empty(shape=3)
    if (Prob_A > Prob_B) : 
        results[0] +=1
    else:
        results[1] +=1
    
    #Accuracy %
    results[2] = (results[0]/(results[0]+results[1]))*100
    np.savetxt('field_as_test_dataset.txt', results, fmt='%f')
    
    #Print out accuracy %
    print ("ProbA={0:.2f},ProbB={1:.2f}->Test image {2} classified as {3}. Total Classification %: Classified as A {4:.2f}%, Classified as B {5:.2f}%".format(Prob_A,Prob_B,test_line,out,results[2],100-results[2]), end="\r")
    
def main():
    for k in range(0, 200):#change test sample count here
        nbc_train_classifier(k)
    print("\n")    

if __name__== "__main__" :
    main()

