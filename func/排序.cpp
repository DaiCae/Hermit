#include<stdio.h>
void pai_xu(int N,double B[]){
        double A=0.0;
        //double *Shu_zu=new double[10]
        for(int i=0;i<N;i++){
                for(int j=0;j<N-i-1;j++){
                        if(B[j]>B[j+1]){
                                A=B[j];
                                B[j]=B[j+1];
                                B[j+1]=A;


                        }
                }
        }

}
int main(){
double A[10]={10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0};
for(int i=0;i<10;i++){
                printf("%lf  \n",A[i]);
}




pai_xu(10,A);
for(int i=0;i<10;i++){
        printf("%lf  \n",A[i]);
}


}