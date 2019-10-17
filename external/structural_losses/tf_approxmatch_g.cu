
__global__ void approxmatch(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,float * __restrict__ match,float * temp){
    const float pi = 3.14159265358979323846;
	float * remainL=temp+blockIdx.x*(n+m)*2, * remainR=temp+blockIdx.x*(n+m)*2+n,*ratioL=temp+blockIdx.x*(n+m)*2+n+m,*ratioR=temp+blockIdx.x*(n+m)*2+n+m+n;
	float multiL,multiR;
	if (n>=m){
		multiL=1;
		multiR=n/m;
	}else{
		multiL=m/n;
		multiR=1;
	}
	const int Block=1024;
	__shared__ float buf[Block*3];
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
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				float x1=0,y1=0;
				if (k<n){
					x1=xyz1[i*n*2+k*2+0];
					y1=xyz1[i*n*2+k*2+1];
				}
				float suml=1e-9f;
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						float x2=xyz2[i*m*2+l0*2+l*2+0];
						float y2=xyz2[i*m*2+l0*2+l*2+1];
						buf[l*3+0]=x2;
						buf[l*3+1]=y2;
						buf[l*3+2]=remainR[l0+l];
					}
					__syncthreads();
					for (int l=0;l<lend;l++){
						float x2=buf[l*3+0];
						float y2=buf[l*3+1];
    float dphi = x2-x1;
    dphi = dphi + pi;
    dphi = fmod(dphi,(2*pi));
    dphi = dphi - pi;
						float d=level*(dphi*dphi+(y2-y1)*(y2-y1));
						float w=__expf(d)*buf[l*3+2];
						suml+=w;
					}
					__syncthreads();
				}
				if (k<n)
					ratioL[k]=remainL[k]/suml;
			}
			/*for (int k=threadIdx.x;k<n;k+=gridDim.x){
				float x1=xyz1[i*n*2+k*2+0];
				float y1=xyz1[i*n*2+k*2+1];
				float suml=1e-9f;
				for (int l=0;l<m;l++){
					float x2=xyz2[i*m*2+l*2+0];
					float y2=xyz2[i*m*2+l*2+1];
    float dphi = x2-x1;
    dphi = dphi + pi;
    dphi = fmod(dphi,(2*pi));
    dphi = dphi - pi;
					float w=expf(level*(dphi*dphi+(y2-y1)*(y2-y1)))*remainR[l];
					suml+=w;
				}
				ratioL[k]=remainL[k]/suml;
			}*/
			__syncthreads();
			for (int l0=0;l0<m;l0+=blockDim.x){
				int l=l0+threadIdx.x;
				float x2=0,y2=0,z2=0;
				if (l<m){
					x2=xyz2[i*m*2+l*2+0];
					y2=xyz2[i*m*2+l*2+1];
				}
				float sumr=0;
				for (int k0=0;k0<n;k0+=Block){
					int kend=min(n,k0+Block)-k0;
					for (int k=threadIdx.x;k<kend;k+=blockDim.x){
						buf[k*3+0]=xyz1[i*n*2+k0*2+k*2+0];
						buf[k*3+1]=xyz1[i*n*2+k0*2+k*2+1];
						buf[k*3+2]=ratioL[k0+k];
					}
					__syncthreads();
					for (int k=0;k<kend;k++){
						float x1=buf[k*3+0];
						float y1=buf[k*3+1];
    float dphi = x2-x1;
    dphi = dphi + pi;
    dphi = fmod(dphi,(2*pi));
    dphi = dphi - pi;
						float w=__expf(level*(dphi*dphi+(y2-y1)*(y2-y1)))*buf[k*3+2];
						sumr+=w;
					}
					__syncthreads();
				}
				if (l<m){
					sumr*=remainR[l];
					float consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
					ratioR[l]=consumption*remainR[l];
					remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
				}
			}
			/*for (int l=threadIdx.x;l<m;l+=blockDim.x){
				float x2=xyz2[i*m*2+l*2+0];
				float y2=xyz2[i*m*2+l*2+1];
				float sumr=0;
				for (int k=0;k<n;k++){
					float x1=xyz1[i*n*2+k*2+0];
					float y1=xyz1[i*n*2+k*2+1];
    float dphi = x2-x1;
    dphi = dphi + pi;
    dphi = fmod(dphi,(2*pi));
    dphi = dphi - pi;
					float w=expf(level*(dphi*dphi+(y2-y1)*(y2-y1)))*ratioL[k];
					sumr+=w;
				}
				sumr*=remainR[l];
				float consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
				ratioR[l]=consumption*remainR[l];
				remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
			}*/
			__syncthreads();
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				float x1=0,y1=0,z1=0;
				if (k<n){
					x1=xyz1[i*n*2+k*2+0];
					y1=xyz1[i*n*2+k*2+1];
				}
				float suml=0;
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						buf[l*3+0]=xyz2[i*m*2+l0*2+l*2+0];
						buf[l*3+1]=xyz2[i*m*2+l0*2+l*2+1];
						buf[l*3+2]=ratioR[l0+l];
					}
					__syncthreads();
					float rl=ratioL[k];
					if (k<n){
						for (int l=0;l<lend;l++){
							float x2=buf[l*3+0];
							float y2=buf[l*3+1];
    float dphi = x2-x1;
    dphi = dphi + pi;
    dphi = fmod(dphi,(2*pi));
    dphi = dphi - pi;
							float w=__expf(level*(dphi*dphi+(y2-y1)*(y2-y1)))*rl*buf[l*3+2];
							match[i*n*m+(l0+l)*n+k]+=w;
							suml+=w;
						}
					}
					__syncthreads();
				}
				if (k<n)
					remainL[k]=fmaxf(0.0f,remainL[k]-suml);
			}
			/*for (int k=threadIdx.x;k<n;k+=blockDim.x){
				float x1=xyz1[i*n*2+k*2+0];
				float y1=xyz1[i*n*2+k*2+1];
				float suml=0;
				for (int l=0;l<m;l++){
					float x2=xyz2[i*m*2+l*2+0];
					float y2=xyz2[i*m*2+l*2+1];
    float dphi = x2-x1;
    dphi = dphi + pi;
    dphi = fmod(dphi,(2*pi));
    dphi = dphi - pi;
					float w=expf(level*(dphi*dphi+(y2-y1)*(y2-y1)))*ratioL[k]*ratioR[l];
					match[i*n*m+l*n+k]+=w;
					suml+=w;
				}
				remainL[k]=fmaxf(0.0f,remainL[k]-suml);
			}*/
			__syncthreads();
		}
	}
}
void approxmatchLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,float * match,float * temp){
	approxmatch<<<32,512>>>(b,n,m,xyz1,xyz2,match,temp);
}
__global__ void matchcost(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ out){
    const float pi = 3.14159265358979323846;
	__shared__ float allsum[512];
	const int Block=1024;
	__shared__ float buf[Block*2];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		float subsum=0;
		for (int k0=0;k0<n;k0+=blockDim.x){
			int k=k0+threadIdx.x;
			float x1=0,y1=0,z1=0;
			if (k<n){
				x1=xyz1[i*n*2+k*2+0];
				y1=xyz1[i*n*2+k*2+1];
			}
			for (int l0=0;l0<m;l0+=Block){
				int lend=min(m,l0+Block)-l0;
				for (int l=threadIdx.x;l<lend*2;l+=blockDim.x)
					buf[l]=xyz2[i*m*2+l0*2+l];
				__syncthreads();
				if (k<n){
					for (int l=0;l<lend;l++){
						float x2=buf[l*2+0];
						float y2=buf[l*2+1];
    float dphi = x2-x1;
    dphi = dphi + pi;
    dphi = fmod(dphi,(2*pi));
    dphi = dphi - pi;
						float d=sqrtf(dphi*dphi+(y2-y1)*(y2-y1));
						subsum+=d*match[i*n*m+(l0+l)*n+k];
					}
				}
				__syncthreads();
			}
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
__global__ void matchcostgrad2(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ grad2){
    const float pi = 3.14159265358979323846;
	__shared__ float sum_grad[256*2];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		int kbeg=m*blockIdx.y/gridDim.y;
		int kend=m*(blockIdx.y+1)/gridDim.y;
		for (int k=kbeg;k<kend;k++){
			float x2=xyz2[(i*m+k)*2+0];
			float y2=xyz2[(i*m+k)*2+1];
			float subsumx=0,subsumy=0,subsumz=0;
			for (int j=threadIdx.x;j<n;j+=blockDim.x){
				float x1=xyz1[(i*n+j)*2+0];
				float y1=y2-xyz1[(i*n+j)*2+1];
    float dphi = x2-x1;
    dphi = dphi + pi;
    dphi = fmod(dphi,(2*pi));
    dphi = dphi - pi;
				float d=match[i*n*m+k*n+j]*rsqrtf(fmaxf(dphi*dphi+y1*y1,1e-20f));
				subsumx+=dphi*d;
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
__global__ void matchcostgrad1(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ grad1){
    const float pi = 3.14159265358979323846;
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int l=threadIdx.x;l<n;l+=blockDim.x){
			float x1=xyz1[i*n*2+l*2+0];
			float y1=xyz1[i*n*2+l*2+1];
			float dx=0,dy=0,dz=0;
			for (int k=0;k<m;k++){
				float x2=xyz2[i*m*2+k*2+0];
				float y2=xyz2[i*m*2+k*2+1];
    float dphi = x1-x2;
    dphi = dphi + pi;
    dphi = fmod(dphi,(2*pi));
    dphi = dphi - pi;
				float d=match[i*n*m+k*n+l]*rsqrtf(fmaxf(dphi*dphi+(y1-y2)*(y1-y2),1e-20f));
				dx+=dphi*d;
				dy+=(y1-y2)*d;
			}
			grad1[i*n*2+l*2+0]=dx;
			grad1[i*n*2+l*2+1]=dy;
		}
	}
}
void matchcostgradLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2){
	matchcostgrad1<<<32,512>>>(b,n,m,xyz1,xyz2,match,grad1);
	matchcostgrad2<<<dim3(32,32),256>>>(b,n,m,xyz1,xyz2,match,grad2);
}

