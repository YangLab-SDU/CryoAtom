#include "ms.h"
#include <ctime>
#include <algorithm>
#include <vector>
using std::vector;

bool msshift(MRC *m, Points *p, float resol, float max_shift, vector<vector<float>>& init_crds){
    time_t tick0 = time(0);

    int i,j,k,ind;
    int cnt=0;
    int xdim=m->mxyz[0];
    int ydim=m->mxyz[1];
    int zdim=m->mxyz[2];
    int xydim=xdim * ydim;


    // Handle init crds
    if (init_crds.empty()){
        printf("# No initial locations are specified, shift on all grids\n");

        if((p->cd=(double **)malloc(sizeof(double *)* (m->Nvoxel + 1000)))==NULL)
            return true;
        if((p->origrid=(int **)malloc(sizeof(int *)*m->Nvoxel))==NULL)
            return true;

        for(int x=0;x<xdim;x++){
            for(int y=0;y<ydim;y++){
                for(int z=0;z<zdim;z++){
                    ind=xydim*z+xdim*y+x;

                    if(m->line[ind]<=0.00)
                        continue;
                    
                    if((p->cd[cnt]=(double *)malloc(sizeof(double)*3))==NULL)
                        return true;
                    if((p->origrid[cnt]=(int *)malloc(sizeof(int)*3))==NULL)
                        return true;
                    
                    //map origin xyz and xwidth based coordinates
                    p->cd[cnt][0]=(double)x;
                    p->cd[cnt][1]=(double)y;
                    p->cd[cnt][2]=(double)z;

                    //Original Grid positions
                    p->origrid[cnt][0]=x;
                    p->origrid[cnt][1]=y;
                    p->origrid[cnt][2]=z;

                    cnt++;
        
                }
            }
        }

        p->Ncd=cnt;
        p->Nori=cnt;
    }else{
        cnt = init_crds.size();
        int sort[3];
        float origin[3];

        printf("# Specify %d initial locations, shift on specified locations\n", cnt);

        // Shift origin by ncrsstart and originxyz
        for(int i = 0; i < 3; ++i){
            sort[i] = m->mapcrs[i] - 1;
            origin[i] = m->originxyz[i] + m->ncrsstart[sort[i]] * m->vsize[i];
            printf("# Origin = %.6f\n", origin[i]);
        }

        // Allocate
        if((p->cd=(double **)malloc(sizeof(double *)*cnt))==NULL)
            return true;
        if((p->origrid=(int **)malloc(sizeof(int *)*cnt))==NULL)
            return true;

        for(int i = 0; i < cnt; ++i){
            if((p->cd[i]=(double *)malloc(sizeof(double)*3))==NULL)
               return true;
            if((p->origrid[i]=(int *)malloc(sizeof(int)*3))==NULL)
               return true;

            //Get original voxel index
            int x = (init_crds[i][0] - origin[0]) / m->vsize[0];
            int y = (init_crds[i][1] - origin[1]) / m->vsize[1];
            int z = (init_crds[i][2] - origin[2]) / m->vsize[2];
            if (x < 0) x = 0; if (x >= xdim) x = xdim - 1;
            if (y < 0) y = 0; if (y >= ydim) y = ydim - 1;
            if (z < 0) z = 0; if (z >= zdim) z = zdim - 1;

            //map origin xyz and xwidth based coordinates
            p->cd[i][0]=(double)x;
            p->cd[i][1]=(double)y;
            p->cd[i][2]=(double)z;

            //Original Grid positions
            p->origrid[i][0]=x;
            p->origrid[i][1]=y;
            p->origrid[i][2]=z;
        }

        p->Ncd=cnt;
        p->Nori=cnt;
    }


    printf("# Start shift\n");

    if((p->dens=(double *)malloc(sizeof(double)*cnt))==NULL)
        return true;
    
    //Setup Filter
    //Gaussian kernel dreso=window size

    double dreso=2.0;
    double c = 1.0;
    double gstep=m->vsize[0];
    double fs=(dreso/gstep)*0.5;
    fs=fs*fs;
    double fsiv=1.000/fs;
    double fmaxd=(dreso/gstep)*2.0;
    double ker;
    double Kernel;
    double C;

    // Set up max shift
    double rshift = max_shift;
    double rshift2 = (rshift / gstep); rshift2 *= rshift2;
    printf("# Max shift = %.4f\n", rshift);

    //using default kernel, default = 2.0
    if(resol <= 4.0)
        dreso = 2.0;
    else if(resol <= 5.0){
        dreso = 2.5;
    }else if(resol <= 6.0){
        dreso = 3.0;
    }else if(resol <= 7.0){
        dreso = 3.5;
    }else if(resol <= 8.0){
        dreso = 4.0;
    }else{
        dreso = 4.5;
    }

    gstep=m->vsize[0];
    fs=(dreso/gstep)*0.5;
    fs=fs*fs;
    fsiv=1.000/fs;
    fmaxd=(dreso/gstep)*2.0;
    Kernel = -1.5 * fsiv;
    C = 1.0;

    printf("# Half window size = %f\n", fmaxd);

    //p->Ncd=cnt;

    //accelerate by multi threads
    #pragma omp parallel for schedule(dynamic, 5)
    for(int i=0;i<cnt;i++){
        int stp[3],endp[3],ind2;
        double pos[3],pos2[3],pos0[3];
        double tmpcd[3];
        double rx,ry,rz,d2;
        double v,dtotal,rd;
        pos0[0]=p->cd[i][0];
        pos0[1]=p->cd[i][1];
        pos0[2]=p->cd[i][2];

        pos[0]=p->cd[i][0];
        pos[1]=p->cd[i][1];
        pos[2]=p->cd[i][2];
        int nshift = 0;
        if(i % 50000 == 0)
            printf("# Accomplished %d / %d : %.2f%%\n", i, cnt, 100.0 * (double)i / cnt);

        //Starting shift
        while(nshift < 5000){
            //Start Point
            ++nshift;
            stp[0]=(int)(pos[0]-fmaxd);
            stp[1]=(int)(pos[1]-fmaxd);
            stp[2]=(int)(pos[2]-fmaxd);

            if(stp[0]<0)stp[0]=0;
            if(stp[1]<0)stp[1]=0;
            if(stp[2]<0)stp[2]=0;

            endp[0]=(int)(pos[0]+fmaxd+1);
            endp[1]=(int)(pos[1]+fmaxd+1);
            endp[2]=(int)(pos[2]+fmaxd+1);

            if(endp[0]>=xdim) endp[0]=xdim;
            if(endp[1]>=ydim) endp[1]=ydim;
            if(endp[2]>=zdim) endp[2]=zdim;

            dtotal=0;
            pos2[0]=pos2[1]=pos2[2]=0;
            for(int xp = stp[0]; xp < endp[0]; xp++){
                rx=(double)xp-pos[0];
                rx=rx*rx;
                for(int yp = stp[1]; yp < endp[1]; yp++){
                    ry=(double)yp-pos[1];
                    ry=ry*ry;
                    for(int zp = stp[2]; zp < endp[2]; zp++){
                        rz=(double)zp-pos[2];
                        rz=rz*rz;
                        d2=rx+ry+rz;
          
                        //d=exp(-1.50*fr*fsiv)*amap(ii,jj,kk)
                        ind2=xydim*zp+xdim*yp+xp;
                        //v=exp(-1.50*d2*fsiv)*m->line[ind2];
                        //v=C*exp(Kernel*d2)*m->line[ind2];
                        v=exp(Kernel*d2)*m->line[ind2];
                        dtotal+=v;
                        if(v>0)
                        //printf("d %f %d %d %d\n",v,xp,yp,zp);
                        pos2[0]+=v*(double)xp;
                        pos2[1]+=v*(double)yp;
                        pos2[2]+=v*(double)zp;
                    }
                }
            }
    
            //If delta shift is too small, break
            if(dtotal<=0.00001)
                break;
    
            rd=1.00/dtotal;
            pos2[0]*=rd;
            pos2[1]*=rd;
            pos2[2]*=rd;
            tmpcd[0]=pos[0]-pos2[0];
            tmpcd[1]=pos[1]-pos2[1];
            tmpcd[2]=pos[2]-pos2[2];

            pos[0]=pos2[0];
            pos[1]=pos2[1];
            pos[2]=pos2[2];
            //printf("*%d %f %f %f v= %f\n",i,pos[0],pos[1],pos[2],dtotal);
            if(tmpcd[0]*tmpcd[0]+tmpcd[1]*tmpcd[1]+tmpcd[2]*tmpcd[2]<0.001)
                break;

            //If exceed max shift
            tmpcd[0] = pos0[0] - pos2[0];
            tmpcd[1] = pos0[1] - pos2[1];
            tmpcd[2] = pos0[2] - pos2[2];
            if(tmpcd[0]*tmpcd[0]+tmpcd[1]*tmpcd[1]+tmpcd[2]*tmpcd[2]>rshift2)
                break;

        }
    
        //Set the shifted coords
        p->cd[i][0]=pos[0];
        p->cd[i][1]=pos[1];
        p->cd[i][2]=pos[2];
        p->dens[i]=dtotal;
    }

    //printf("%8.3f %8.3f %8.3f\n", p->cd[0][0], p->cd[0][1], p->cd[0][2]);

    printf("# End shift\n");
    time_t tick1 = time(0);
    printf("# Time consumption = %.2f\n", double(tick1 - tick0));

    return false;
}



bool msmerge(MRC *m, Points *p, double filter, double distance){
    time_t tick0 = time(0);
    double mpix = distance; //default = 0.5
    double dcut= mpix/m->vsize[0];
    double d2cut=dcut*dcut;
    double rdcut=filter;
    double dmax,dmin,drange;
    bool *stock;
    int *tmp_member,*tmp_member2;

    //Allocate memory
    if((stock=(bool *)malloc(sizeof(bool)*p->Ncd))==NULL)
        return true;
    if((tmp_member=(int *)malloc(sizeof(int)*p->Ncd))==NULL)
        return true;
    if((tmp_member2=(int *)malloc(sizeof(int)*p->Ncd))==NULL)
        return true;

    dmax=0;
    dmin=999999.99;
 
    //find min/max density
    for(int i=0;i<p->Ncd;i++){
        if(p->dens[i]<dmin) dmin=p->dens[i];
        if(p->dens[i]>dmax) dmax=p->dens[i];
    }

    drange=dmax-dmin; //density range
    p->dmax = dmax;
    p->dmin = dmin;
    double rv_range=1.00/drange; 
    printf("# Density max = %.2f min = %.2f\n",dmax,dmin);
    printf("# Starting to merge points\n");
    
    if((p->member=(int *)malloc(sizeof(int)*p->Ncd))==NULL)
        return true;
    for(int i=0;i<p->Ncd;i++)
        p->member[i] = i; //member : i'th members
    for(int i=0;i<p->Ncd;i++)
        stock[i] = true;  //initialize stock table

    //#pragma omp parallel for schedule(dynamic,5)
    for(int i=0;i<p->Ncd-1;i++){
        double tmp[3], d2;
        if((p->dens[i]-dmin)*rv_range < rdcut){
            //discard points with normalized density ratio below rdcut
            //printf("%d %.4f %.4f\n", i, (p->dens[i]-dmin)*rv_range, rdcut);
            stock[i]=false;
        }

        if(stock[i]==false)
            continue;
        
        for(int j=i+1;j<p->Ncd;j++){
            if(stock[j]==false)
                continue;

            tmp[0]=p->cd[i][0]-p->cd[j][0]; //calculate distance
            tmp[1]=p->cd[i][1]-p->cd[j][1];
            tmp[2]=p->cd[i][2]-p->cd[j][2];
            d2=tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2];

            if(d2 < d2cut){
                //printf("%d %d %.4f %.4f\n", i, j, d2, d2cut);
                //Keep point with higher density
                if(p->dens[i]>p->dens[j]){
                    stock[j]=false;  //all p->member[i] are initialized to i
                    p->member[j]=i; //if j is not stocked, p->member[j] = i, j belongs to cluster i
                }else{
                    stock[i]=false;
                    p->member[i]=j; 
                    break;
                }
            }
        }
    }

    //Update member data using disjoint set
    for(int i=0;i<p->Ncd;i++){ //for i
        int now=p->member[i];   //now : the cluster that i belongs to
        //find the root of now (same as how disjoint set does)
        for(int j=0;j<p->Ncd;j++){ //for another j
            if(now==p->member[now])  //if now belongs to itself, then break
                break;
            
            now=p->member[now]; //record the cluster now belongs to
        }
        p->member[i]=now;
    }
    
    int Nmerge=0; //merging
    int sort[3];
    double origin[3];

    // Shift origin by ncrsstart and originxyz
    for(int i = 0; i < 3; ++i){
        sort[i] = m->mapcrs[i] - 1;
        origin[i] = m->originxyz[i] + m->ncrsstart[sort[i]] * m->vsize[i];
        printf("# Origin = %.6f\n", origin[i]);
    }

    std::vector<std::vector<double>> coords;
    coords.reserve(5000);
    for(int i=0;i<p->Ncd;i++){
        if(stock[i]){ //if i is reserved
            p->cd[Nmerge][0]=p->cd[i][0];  //adjust i into range [0, Nmerge)
            p->cd[Nmerge][1]=p->cd[i][1];
            p->cd[Nmerge][2]=p->cd[i][2];
            p->dens[Nmerge]=p->dens[i];
    
            coords.push_back(std::vector<double>(3, 0.0));
            coords[Nmerge][0] = p->cd[i][0] * m->vsize[0] + origin[0];// + m->ncrsstart[sort[0]] * m->vsize[0];
            coords[Nmerge][1] = p->cd[i][1] * m->vsize[1] + origin[1];// + m->ncrsstart[sort[1]] * m->vsize[1];
            coords[Nmerge][2] = p->cd[i][2] * m->vsize[2] + origin[2];// + m->ncrsstart[sort[2]] * m->vsize[2];

            //No adding origin shift?
            /*
            coords.push_back(vector<double>(3));
            coords[Nmerge][0] = p->cd[i][0] * m->vsize[0];
            coords[Nmerge][1] = p->cd[i][1] * m->vsize[1];
            coords[Nmerge][2] = p->cd[i][2] * m->vsize[2];
            */

            tmp_member[i]=Nmerge; //save cluster that i belongs to
            Nmerge++;
        }else{ //if i is not reserved
            tmp_member[i]=-1; //mark to -1
        }
    }
    //save the coords to Points*
    p->coords.swap(coords);

    for(int i=0;i<p->Ncd;i++)
        tmp_member2[i] = tmp_member[p->member[i]]; //i's root's root
    for(int i=0;i<p->Ncd;i++)
        p->member[i] = tmp_member2[i]; //update
 
    printf("# After Merge %d points reserved\n", Nmerge);
    p->Ncd  = Nmerge;
    p->Np   = Nmerge;
    time_t tick1 = time(0);
    return false;
}

