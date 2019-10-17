//n<=4096, m<=1024
__global__ void approxmatch(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,float * __restrict__ match){
	const int MaxN=4096,MaxM=1024;
	__shared__ float remainL[MaxN],remainR[MaxM],ratioR[MaxM],ratioL[MaxN];
	__shared__ int listR[MaxM],lc;
	float multiL,multiR;
	if (n>=m){
		multiL=1;
		multiR=n/m;
	}else{
		multiL=m/n;
		multiR=1;
	}
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x;j<n*m;j+=blockDim.x)
			match[i*n*m+j]=0;
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			remainL[j]=multiL;
		for (int j=threadIdx.x;j<m;j+=blockDim.x)
			remainR[j]=multiR;
		__syncthreads();
		for (int j=7;j>=-2;j--){
			float level=-powf(4.0f,j);
			if (j==-2){
				level=0;
			}
			if (threadIdx.x==0){
				lc=0;
				for (int k=0;k<m;k++)
					if (remainR[k]>0)
						listR[lc++]=k;
			}
			__syncthreads();
			int _lc=lc;
			for (int k=threadIdx.x;k<n;k+=blockDim.x){
				float suml=1e-9f;
				float x1=xyz1[(i*n+k)*2+0];
				float y1=xyz1[(i*n+k)*2+1];
				//for (int l=0;l<m;l++){
				for (int _l=0;_l<_lc;_l++){
					int l=listR[_l];
					float x2=xyz2[(i*m+l)*2+0]-x1;
					float y2=xyz2[(i*m+l)*2+1]-y1;
					float w=expf(level*(x2*x2+y2*y2))*remainR[l];
					suml+=w;
				}
				ratioL[k]=remainL[k]/suml;
			}
			__syncthreads();
			//for (int k=threadIdx.x;k<m;k+=blockDim.x){
			for (int _k=threadIdx.x;_k<lc;_k+=blockDim.x){
				int k=listR[_k];
				float sumr=0;
				float x2=xyz2[(i*m+k)*2+0];
				float y2=xyz2[(i*m+k)*2+1];
				for (int l=0;l<n;l++){
					float x1=xyz1[(i*n+l)*2+0]-x2;
					float y1=xyz1[(i*n+l)*2+1]-y2;
					float w=expf(level*(x1*x1+y1*y1))*ratioL[l];
					sumr+=w;
				}
				sumr*=remainR[k];
				float consumption=fminf(remainR[k]/(sumr+1e-9f),1.0f);
				ratioR[k]=consumption*remainR[k];
				remainR[k]=fmaxf(0.0f,remainR[k]-sumr);
			}
			__syncthreads();
			for (int k=threadIdx.x;k<n;k+=blockDim.x){
				float suml=0;
				float x1=xyz1[(i*n+k)*2+0];
				float y1=xyz1[(i*n+k)*2+1];
				for (int _l=0;_l<_lc;_l++){
					int l=listR[_l];
					float x2=xyz2[(i*m+l)*2+0]-x1;
					float y2=xyz2[(i*m+l)*2+1]-y1;
					float w=expf(level*(x2*x2+y2*y2))*ratioL[k]*ratioR[l];
					match[i*n*m+l*n+k]+=w;
					suml+=w;
				}
				remainL[k]=fmaxf(0.0f,remainL[k]-suml);
			}
			__syncthreads();
		}
	}
}
void approxmatchLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,float * match){
	approxmatch<<<32,512>>>(b,n,m,xyz1,xyz2,match);
}
__global__ void matchcost(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ out){
	__shared__ float allsum[512];
	const int Block=256;
	__shared__ float buf[Block*2];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		float subsum=0;
		for (int k0=0;k0<m;k0+=Block){
			int endk=min(m,k0+Block);
			for (int k=threadIdx.x;k<(endk-k0)*2;k+=blockDim.x){
				buf[k]=xyz2[i*m*2+k0*2+k];
			}
			__syncthreads();
			for (int j=threadIdx.x;j<n;j+=blockDim.x){
				float x1=xyz1[(i*n+j)*2+0];
				float y1=xyz1[(i*n+j)*2+1];
				for (int k=0;k<endk-k0;k++){
					//float x2=xyz2[(i*m+k)*2+0]-x1;
					//float y2=xyz2[(i*m+k)*2+1]-y1;
					float x2=buf[k*2+0]-x1;
					float y2=buf[k*2+1]-y1;
					float d=sqrtf(x2*x2+y2*y2);
					subsum+=match[i*n*m+(k0+k)*n+j]*d;
				}
			}
			__syncthreads();
		}
		allsum[threadIdx.x]=subsum;
		for (int j=1;j<blockDim.x;j<<=1){
			__syncthreads();
			if ((threadIdx.x&j)==0 && threadIdx.x+j<blockDim.x){
				allsum[threadIdx.x]+=allsum[threadIdx.x+j];
			}
		}
		if (threadIdx.x==0)
			out[i]=allsum[0];
		__syncthreads();
	}
}
void matchcostLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * out){
	matchcost<<<32,512>>>(b,n,m,xyz1,xyz2,match,out);
}
__global__ void matchcostgrad(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * grad2){
	__shared__ float sum_grad[256*2];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		int kbeg=m*blockIdx.y/gridDim.y;
		int kend=m*(blockIdx.y+1)/gridDim.y;
		for (int k=kbeg;k<kend;k++){
			float x2=xyz2[(i*m+k)*2+0];
			float y2=xyz2[(i*m+k)*2+1];
			float subsumx=0,subsumy=0,subsumz=0;
			for (int j=threadIdx.x;j<n;j+=blockDim.x){
				float x1=x2-xyz1[(i*n+j)*2+0];
				float y1=y2-xyz1[(i*n+j)*2+1];
				float d=match[i*n*m+k*n+j]/fmaxf(sqrtf(x1*x1+y1*y1),1e-20f);
				subsumx+=x1*d;
				subsumy+=y1*d;
			}
			sum_grad[threadIdx.x*2+0]=subsumx;
			sum_grad[threadIdx.x*2+1]=subsumy;
			for (int j=1;j<blockDim.x;j<<=1){
				__syncthreads();
				int j1=threadIdx.x;
				int j2=threadIdx.x+j;
				if ((j1&j)==0 && j2<blockDim.x){
					sum_grad[j1*2+0]+=sum_grad[j2*2+0];
					sum_grad[j1*2+1]+=sum_grad[j2*2+1];
				}
			}
			if (threadIdx.x==0){
				grad2[(i*m+k)*2+0]=sum_grad[0];
				grad2[(i*m+k)*2+1]=sum_grad[1];
			}
			__syncthreads();
		}
	}
}
void matchcostgradLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad2){
	matchcostgrad<<<dim3(32,32),256>>>(b,n,m,xyz1,xyz2,match,grad2);
}

