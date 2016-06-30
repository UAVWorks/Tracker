/*
 * FerNNClassifier.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */

#include "FerNNClassifier.h"
using namespace cv;
using namespace std;

void FerNNClassifier::read(const FileNode& file){
  ///Classifier Parameters
  valid = (float)file["valid"];
  ncc_thesame = (float)file["ncc_thesame"];
  nstructs = (int)file["num_trees"];//��ľ����һ�������鹹����ÿ����������ͼ���Ĳ�ͬ��ͼ��ʾ���ĸ���  
  structSize = (int)file["num_features"]; //ÿ����������������Ҳ��ÿ�����Ľڵ����������ÿһ����������Ϊһ�����߽ڵ�  
  thr_fern = (float)file["thr_fern"];
  thr_nn = (float)file["thr_nn"];
  thr_nn_valid = (float)file["thr_nn_valid"];
}
//׼�������� ��Ҫ��ʼ�� ���Ϸ�����ģ��
void FerNNClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  int totalFeatures = nstructs*structSize;//10*13
  //��ά���� ����ȫ���߶ȣ�scales����ɨ�贰�ڣ�ÿ���߶Ȱ���totalFeatures������
  features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));
  RNG& rng = theRNG();	//�̶��÷���opencv�Դ���һ�������������
  float x1f,x2f,y1f,y2f;//��С��1�������
  int x1, x2, y1, y2;
  //���Ϸ���������n��������������ÿ�����������ǻ���һ��pixel comparisons�����رȽϼ����ģ�  
  //pixel comparisons�Ĳ�������������һ����һ����patchȥ��ɢ�����ؿռ䣬�������п��ܵĴ�ֱ��ˮƽ��pixel comparisons  
  //Ȼ�����ǰ���Щpixel comparisons��������n����������ÿ���������õ���ȫ��ͬ��pixel comparisons���������ϣ���  
  //���������з�������������ͳһ�����Ϳ��Ը�������patch��  
  //�������ȥ���ÿһ���߶�ɨ�贰�ڵ�����  
  for (int i=0;i<totalFeatures;i++){
      x1f = (float)rng;
      y1f = (float)rng;
      x2f = (float)rng;
      y2f = (float)rng;
	// cout << "x1f=" << x1f << "y1f=" << y1f << "x2f=" << x2f << "y2f=" << y2f << endl;
      for (int s=0;s<scales.size();s++){
          x1 = x1f * scales[s].width;
          y1 = y1f * scales[s].height;
          x2 = x2f * scales[s].width;
          y2 = y2f * scales[s].height;
		  //��s�ֳ߶ȵĵ�i����������ԡ���ע�ⲻ������ֵ��  ���������������ص����� 
          features[s][i] = Feature(x1, y1, x2, y2);
		//  cout << "x1=" << x1 << "y1=" << y1 << "x2=" << x2 << "y2=" << y2 << endl;
      }

  }
  //Thresholds
  thrN = 0.5*nstructs;	//Ŀ����Ǹ�����ֵ

  //Initialize Posteriors  ��ʼ���������  
  //�������ָÿһ���������Դ����ͼ��Ƭ�������ضԱȣ�ÿһ�����ضԱȵõ�0����1�����е�����13��comparison�Աȣ�  
  //����һ��13λ�Ķ����ƴ���x��Ȼ��������һ����¼�˺�����ʵ�����P(y|x)��yΪ0����1�������ࣩ��Ҳ���ǳ���x��  
  //�����ϣ���ͼ��ƬΪy�ĸ����Ƕ��ٶ�n�������������ĺ��������ƽ��������0.5���ж��京��Ŀ��  

	  //ÿһ��ÿ����ά��һ��������ʵķֲ�������ֲ���2^d����Ŀ��entries��������d�����رȽ�pixel comparisons  
	  //�ĸ�����������structSize����13��comparison�����Ի����2^13��8,192�����ܵ�code��ÿһ��code��Ӧһ���������  
	  //�������P(y|x)= #p/(#p+#n) ,#p��#n�ֱ������͸�ͼ��Ƭ����Ŀ��Ҳ���������pCounter��nCounter  
	  //��ʼ��ʱ��ÿ��������ʶ��ó�ʼ��Ϊ0������ʱ�������淽ʽ���£���֪����ǩ��������ѵ��������ͨ��n��������  
	  //���з��࣬���������������ô��Ӧ��#p��#n�ͻ���£�����P(y|x)Ҳ��Ӧ������
  for (int i = 0; i<nstructs; i++) {
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  }
}

//�ú����õ������image���������Ľڵ㣬Ҳ�����������������13λ�Ķ����ƴ��룩
void FerNNClassifier::getFeatures(const cv::Mat& image,const int& scale_idx, vector<int>& fern){
  int leaf;
  for (int t=0;t<nstructs;t++){//10����
      leaf=0;
      for (int f=0; f<structSize; f++)
	  {//��ʾÿ���������ĸ��� 13 
		  //struct Feature �����ṹ����һ����������� bool operator ()(const cv::Mat& patch) const  
		  //���ص�patchͼ��Ƭ��(y1,x1)��(y2, x2)������رȽ�ֵ������0����1  
		  //Ȼ��leaf�ͼ�¼����13λ�Ķ����ƴ��룬��Ϊ����  
          leaf = (leaf << 1) + features[scale_idx][t*structSize+f](image);//image��ĳ��good_box ����Ӧ�ĳ߶������г�ʼ�����˵���������㡣��ʱ�Ż����Feature�ģ������أ�����ÿƬҶ�Ӷ����������������Ϊ�Ƚϣ����ص��ǵ�һ���������ֵ�Ƿ���ڵڶ����������ֵ����١���������Ϊ����ֵ������leaf�����ܳ
      }
      fern[t]=leaf;//leafΪ�����Ʊ�������
  }
}

float FerNNClassifier::measure_forest(vector<int> fern) {
  float votes = 0;
  for (int i = 0; i < nstructs; i++) {
	  // �������posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]); 
      votes += posteriors[i][fern[i]];
  }
  return votes;
}

//����������������ͬʱ���º������ 
void FerNNClassifier::update(const vector<int>& fern, int C, int N) {
  int idx;
  for (int i = 0; i < nstructs; i++)
  {
      idx = fern[i];
      (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
      if (pCounter[i][idx]==0) 
	  {
          posteriors[i][idx] = 0;
      } 
	  else 
	  {
          posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      }
  }
}

//ѵ�����Ϸ�������n���������������ϣ���������Ժ������posteriors��һ��������ֵ0������Ĭ�������������Ǹ�����������ѵ����
//��һ���ܶ�ε�ѭ�����ܴ𵽱ȽϺõ�Ч�������Կ���posteriors��i����j�������������ֵΪj�������ڵ�i�����ĺ������
//j��2^13�����Ŷ����pcounter��ncounter���ƣ����������ʵ������һ����ֵ���������������������������ֵ��ע�����
//��pcounter��ncounter��posteriors���໥Ӱ��ģ�������Ҫ������ѵ�������űȽϾ�ȷ�����ǿ���������һ������
//����һ��ͼ�����ԣ��������Ŀ������Ƚ�Զ�Ļ�����������һֱ0�������ܶ���Ժ�ͻᷢ�����ǵ�����ֵ�������ncounter
//��ܶ࣬����Ŀ�������ڵ�ͼ������pcounter�ܶ࣬�����ؾ����ֿ����ˣ�����Ե����ȡ�������Ǹ����Ĳ���thr_fern������������ܹ�
//ͨ��ѧϰģ�鲻�ϵ����ģ����������жϵ�Ч����ȽϺá�
void FerNNClassifier::trainF(const vector<std::pair<vector<int>,int> >& ferns,int resample){
  // Conf = function(2,X,Y,Margin,Bootstrap,Idx)
  //                 0 1 2 3      4         5
  //  double *X     = mxGetPr(prhs[1]); -> ferns[i].first
  //  int numX      = mxGetN(prhs[1]);  -> ferns.size()
  //  double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
  //  double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
  //  int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
//thr_fern: 0.6 thrP����ΪPositive thershold  
  thrP = thr_fern*nstructs;                                                          // int step = numX / 10;
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index
                                                            //       double *x = X+nTREES*I; //tree index
           if(ferns[i].second==1){  //Ϊ1��ʾ������                          //       if (Y[I] == 1) {
              if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
				  //measure_forest������������������������ֵ��Ӧ�ĺ�������ۼ�ֵ  
				  //���ۼ�ֵ���С����������ֵ��Ҳ���������������������ȴ������ɸ�������  
				  //���ַ���������ԾͰѸ�������ӵ��������⣬ͬʱ���º������  
                update(ferns[i].first,1,1);                 //             update(x,1,1);
          }else{                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)  0.5
                update(ferns[i].first,0,1);                 //             update(x,0,1);
          }
      }
  //}
}

//ѵ������ڷ�������������һ��ǳ���Ҫ���������������ѵ��nn�������ģ����Ǽ�⣬����������ϵ����б�ǩ���Ƕ�����֪��
//���������ڷ�����������ʱ�������ǩ��1���������������ǵó���������ƶ�<��ֵ��˵�����ǵ�����������׼ȷ����Ҫ����
//�Ͱ������������������������ȥ����������Ҳ�����Ƶ������������ƶ���NNconf��������п��Կ�������pex,nex�Ǿ�����ص�
//������pex��nex��Ӱ����������ƶȣ�����˵�ں����Ĺ����У����ϸ���ѵ�����Ըı�pex,nex�����Ҵ�����train����������ѵ����
//��ֻ�е�һ��ͼƬ������Ϊ�����������ڲ��ϸ��µĹ����У�����ֻ���ʼ��best_boxes�ͺ������Ƿ��ַ������ģ��������״�ģ�
//�Ž�pex����Ϊ������������ΪpexԽ�٣���������ִ��Ч��Խ�ߣ���������Խ�٣������Ա����ϸ���ơ�
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples,const vector<cv::Mat>& nn_images){
  float conf,dummy,mconf;
  vector<int> y(nn_examples.size(),0);
  y[0]=1;//����˵������trainNN������������nn_data��������ֻ��һ��pEx����nn_data[0] 
  vector<int> isin;
  int detaP = 0;
  int detaN = 0;
  for (int i=0;i<nn_examples.size();i++)
  {                          //  For each example
	  //��������ͼ��Ƭ������ģ��֮���������ƶ�conf  
      NNConf(nn_examples[i],isin,conf,dummy,mconf);      //http://blog.csdn.net/mydear_11000/article/details/47946809                //  Measure Relative similarity
	  /*if (y[i] == 1)
		  cout << "��ǰ���ٿ����ƶ�Ϊ"<<conf << endl;*/
      if (y[i]==1 && conf<=thr_nn){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65
		  //thr_nn: 0.65 ��ֵ  
		  //��ǩ�������������������ƶ�С��0.65 ������Ϊ�䲻����ǰ��Ŀ�꣬Ҳ���Ƿ�������ˣ���ʱ��Ͱ����ӵ���������  
          if (isin[1]<0){                                          //      if isnan(isin(2))
              pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
              continue;                                            //        continue;
          }                                                        //      end
          //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
		  /*cout << "modelflag is" << model_flag << endl;*/
		  if (model_flag)
		  {
			  if (conf / mconf < 1.5)
			  {
				  pEx.push_back(nn_examples[i]);
				 /* char str[30];
				  sprintf(str, "%s%d%s", "pEx", pEx.size(), ".png");
				  imwrite(str, nn_images[i]);
				  cout << "����Ŀ��ģ��" << endl;*/
			  }
			  else
			  {
				 /* cout << "������pEx" << endl;*/
			  }
		  }
		  else
		  {
			 /* cout << "������Ŀ��ģ��" << endl;*/
		  }
		  detaP++;
      }                                                            //    end
	  if (y[i] == 0 && conf>0.5)
	  {//  if y(i) == 0 && conf1 > 0.5
		  nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];
		  detaN++;
		/*  char str1[30];
		  sprintf(str1, "%s%d%s","nEx", nEx.size(), ".png");
		  imwrite(str1, nn_images[i]);*/
	  }
  }                                                                 //  end
  acum++;
 // printf("%d. %d������������example����������ڷ�����ģ��: %d ���������� %d�������� �ܹ���%d����������%d��������", acum, nn_examples.size(),detaP,detaN,(int)pEx.size(), (int)nEx.size());
}                                                                  //  end


void FerNNClassifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf,float& msconf){
  /*Inputs:
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
   */
	isin = vector<int>(3, -1); //vector<T> v3(n, i); v3����n��ֵΪi��Ԫ�ء� ����Ԫ�ض���-1  
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
      return;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return;
  }
  vector<float> nccsim;
  Mat ncc(1,1,CV_32F);
  float nccP,csmaxP,maxP=0;
  bool anyP=false;
  int maxPidx,validatedPart = ceil(pEx.size()*valid);//ceil���ش��ڻ��ߵ���ָ�����ʽ����С����  
  int mid3 = ceil(pEx.size()*0.3f)-1;
  int mid7 = ceil(pEx.size()*0.7f);
  float msmaxp=0;
  //float midmaxp=0;
  float nccN, maxN=0;
  bool anyN=false;
  //�Ƚ�ͼ��Ƭp������ģ��M�ľ��루���ƶȣ���������������������ƶȣ�Ҳ���ǽ������ͼ��Ƭ��  
  //����ģ�������е�ͼ��Ƭ����ƥ�䣬�ҳ������Ƶ��Ǹ�ͼ��Ƭ��Ҳ�������ƶȵ����ֵ  
  for (int i = 0; i < pEx.size(); i++)
  {
	  matchTemplate(pEx[i], example, ncc, CV_TM_CCORR_NORMED);      // measure NCC to positive examples
	  nccP = (((float*)ncc.data)[0] + 1)*0.5;//����ƥ�����ƶ� 
	  nccsim.push_back(nccP);
	  if (nccP>ncc_thesame)//ncc_thesame: 0.95  
		  anyP = true;
	  if (nccP > maxP)
	  { //��¼�������ƶ��Լ���Ӧ��ͼ��Ƭindex����ֵ
		  maxP = nccP;
		  maxPidx = i;
		  if (i<validatedPart)
			  csmaxP = maxP;
	  }
	  if (i >= mid3&&i <= mid7)
	  {
		  if (nccP > msmaxp)
		  {
			  msmaxp = nccP;
		  }
	  }
  }


  if (pEx.size() >= 3)
  {
	  for (int i = 0; i < pEx.size()-1; i++)
	  {
		  
		  for (int j = i+1; j < pEx.size(); j++)
		  {
			  if (nccsim[i] < nccsim[j])
			  {
				  float max = nccsim[j];
				  nccsim[j] = nccsim[i];
				  nccsim[i] = max;
			  }

		  }
	  }
	 /* cout << "��һ������ƶ�Ϊ" << nccsim[0] << endl;
	  cout << "�ڶ�������ƶ�Ϊ" << nccsim[1] << endl;
	  cout << "��ֵ���ƶ�Ϊ" << nccsim[pEx.size() / 2] << endl;*/
	  if ((nccsim[0] / nccsim[pEx.size()/2]) > 1.3)
		  model_flag = false;
	  else
		  model_flag = true;
	/*  if (nccsim[0] / nccsim[1] > 1.5)
	  {
		  maxP = nccsim[1];
		  vector<Mat>::iterator it = pEx.begin() + maxPidx;
		  pEx.erase(it);
		  cout << "ɾ��pEX�еĴ���ģ��" << endl;
	  }*/
  }

  //���㸺������������ƶ� 
  for (int i=0;i<nEx.size();i++){
      matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);     //measure NCC to negative examples
      nccN=(((float*)ncc.data)[0]+1)*0.5;
      if (nccN>ncc_thesame)
        anyN=true;
      if(nccN > maxN)
        maxN=nccN;
  }
  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  //Measure Relative Similarity
  //������ƶ� = ��������������ƶ� / ����������������ƶ� + ��������������ƶȣ� 
  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
  dP = 1 - msmaxp;
  msconf = (float)dN / (dN + dP);
 /* if (maxP + maxN == 0)
	  rsconf = 0;
  else
	  rsconf = (float)(maxP / (maxN + maxP));
  if (csmaxP + maxN == 0)
	  csconf = 0;
  else
	  csconf = (float)(csmaxP / (maxN + csmaxP));*/
}
//void FerNNClassifier::NNConf1(const Mat& example, vector<int>& isin, float& rsconf, float& csconf)
//{
//	isin = vector<int>(3, -1); //vector<T> v3(n, i); v3����n��ֵΪi��Ԫ�ء� ����Ԫ�ض���-1  
//	if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
//		rsconf = 0; //    conf1 = zeros(1,size(x,2));
//		csconf = 0;
//		return;
//	}
//	Mat ncc(1, 1, CV_32F);
//	float nccP=0, csminP=0, minP = 1;
//	int minPidx, validatedPart = ceil(pEx.size()*valid)-1;//ceil���ش��ڻ��ߵ���ָ�����ʽ����С����  
//	//�Ƚ�ͼ��Ƭp������ģ��M�ľ��루���ƶȣ���������������������ƶȣ�Ҳ���ǽ������ͼ��Ƭ��  
//	//����ģ�������е�ͼ��Ƭ����ƥ�䣬�ҳ������Ƶ��Ǹ�ͼ��Ƭ��Ҳ�������ƶȵ����ֵ  
//	for (int i = 0; i<pEx.size(); i++)
//	{
//		matchTemplate(pEx[i], example, ncc, CV_TM_CCORR_NORMED);      // measure NCC to positive examples
//		nccP = (((float*)ncc.data)[0] + 1)*0.5;//����ƥ�����ƶ�  
//		if (nccP < minP)
//		{ //��¼�������ƶ��Լ���Ӧ��ͼ��Ƭindex����ֵ
//			minP = nccP;
//			minPidx = i;
//			if (i>=validatedPart)
//				csminP = minP;
//		}
//		rsconf = minP;
//		csconf = csminP;
//	}
//	
//}
void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT,const vector<cv::Mat>& nExT){
float fconf;
//thr_fern = 0;
//thr_nn = 0;
  for (int i=0;i<nXT.size();i++){
	  //���л����������ĺ�����ʵ�ƽ��ֵ�������thr_fern������Ϊ����ǰ��Ŀ��  
	  //measure_forest���ص������к�����ʵ��ۼӺͣ�nstructs Ϊ���ĸ�����Ҳ���ǻ�������������Ŀ ����  
    fconf = (float) measure_forest(nXT[i].first)/nstructs;
    if (fconf>thr_fern)//thr_fern: 0.6 thrP����ΪPositive thershold  
      thr_fern=fconf;//ȡ���ƽ��ֵ��Ϊ �ü��Ϸ������� �µ���ֵ�������ѵ������ 
}
  vector <int> isin;
  float conf,dummy,mconf;
  for (int i=0;i<nExT.size();i++){
      NNConf(nExT[i],isin,conf,dummy,mconf);
      if (conf>thr_nn)
        thr_nn=conf;//ȡ������������ƶ���Ϊ ������ڷ������� �µ���ֵ�������ѵ������  
  }
  if (thr_nn>thr_nn_valid)//��ʼ��ʱΪ0.7
    thr_nn_valid = thr_nn;
}

//���������⣨����ģ�ͣ�������������������ʾ�ڴ�����
void FerNNClassifier::show()
{
  Mat examples((int)pEx.size()*pEx[0].rows,pEx[0].cols,CV_8U);
  double minval;
  Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
  for (int i=0;i<pEx.size();i++)
  {
	  //minMaxLocѰ�Ҿ���һά���鵱����������Mat���壩����Сֵ�����ֵ��λ��.
    minMaxLoc(pEx[i],&minval);//Ѱ��pEx[i]����Сֵ 
    pEx[i].copyTo(ex);
	ex = ex - minval; //������������С����������Ϊ0���������ذ�������Ϊָ������span����һ���µľ���ͷ��  
    //Mat Mat::rowRange(const Range& r) const   //Range �ṹ��������ʼ����ֹ������ֵ��  
    Mat tmp = examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
    ex.convertTo(tmp,CV_8U);
  }
  imshow("Examples",examples);
}
