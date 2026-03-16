
#include "mrcio.h"
using std::abs;

int get_mode(int* mapcrs){
    int mapc = mapcrs[0],
        mapr = mapcrs[1],
        maps = mapcrs[2];

    int ordermode = 0;
    if(mapc == 1 && mapr == 2 && maps == 3){
        ordermode = 1;
    }else if(mapc == 1 && mapr == 3 && maps == 2){
        ordermode = 2;
    }else if(mapc == 2 && mapr == 1 && maps == 3){
        ordermode = 3;
    }else if(mapc == 2 && mapr == 3 && maps == 1){
        ordermode = 4;
    }else if(mapc == 3 && mapr == 1 && maps == 2){
        ordermode = 5;
    }else if(mapc == 3 && mapr == 2 && maps == 1){
        ordermode = 6;
    }else{
        printf("# wrong ordermode.\n");
        exit(1);
    }
    return ordermode;
}

int get_index(int ordermode, int count, int nc, int nr, int ns){
    unsigned ic, ir, is;
    unsigned long ncr, q;

    ncr = nc*nr;
    is = count / ncr;
    q = count - is*ncr;
    ir = q / nc;
    ic = q - ir*nc;

    switch(ordermode) {
        case 1:
            return ic+ir*nc+is*nc*nr;
        case 2:
            return ic+is*nc+ir*nc*ns;
        case 3:
            return ir+ic*nr+is*nr*nc;
        case 4:
            return is+ic*ns+ir*ns*nc;
        case 5:
            return ir+is*nr+ic*nr*ns;
        case 6:
            return is+ir*ns+ic*ns*nr;
        default:
            printf("# illegal ordermode\n");
            exit(1);
    }

}
    
bool check_orthogonal(float* cellb){
    if(abs(cellb[0] - 90.0) > 1e-3 ||
       abs(cellb[1] - 90.0) > 1e-3 ||
       abs(cellb[2] - 90.0) > 1e-3) return false;

    if(abs(cellb[0] - cellb[1]) > 1e-3 ||
       abs(cellb[0] - cellb[2]) > 1e-3 ||
       abs(cellb[1] - cellb[2]) > 1e-3) return false;

    return true;
};

void read_mrc(const char* file, MRC* mrc){
    std::ifstream fin(file, std::ios::binary);
    if(fin.fail()){
      std::cout << "# can't open file : " << file << std::endl;
      exit(1);
    }
    std::cout << "# reading map file : " << file << std::endl;
    fin.read((char*)mrc->ncrs,      sizeof(int) * 3);
    fin.read((char*)&mrc->mode,     sizeof(int));
    fin.read((char*)mrc->ncrsstart, sizeof(int) * 3);
    fin.read((char*)mrc->mxyz,      sizeof(int) * 3);
    fin.read((char*)mrc->cella,     sizeof(float) * 3);
    fin.read((char*)mrc->cellb,     sizeof(float) * 3);
    fin.read((char*)mrc->mapcrs,    sizeof(int) * 3);
    fin.read((char*)&mrc->dmin,     sizeof(float));
    fin.read((char*)&mrc->dmax,     sizeof(float));
    fin.read((char*)&mrc->dmean,    sizeof(float));
    fin.read((char*)&mrc->ispg,     sizeof(int));
    fin.read((char*)&mrc->nsymbt,   sizeof(int));
    fin.read((char*)mrc->extra,     sizeof(int) * 25);
    fin.read((char*)mrc->originxyz, sizeof(float) * 3);
    fin.read((char*)mrc->map,       sizeof(char) * 4);
    fin.read((char*)mrc->machst,    sizeof(char) * 4);
    fin.read((char*)&mrc->rms,      sizeof(float));
    fin.read((char*)&mrc->nlabel,   sizeof(int));
    fin.read((char*)mrc->label,     sizeof(char) * 10 * 80);
    assert(fin.tellg() == 1024); 
    std::cout << "# reaching " << fin.tellg() << " bytes." << std::endl;
    if(mrc->nsymbt > 0){
        std::cout << "# reading extended header." << std::endl;
        mrc->sym = new char[mrc->nsymbt];
        fin.read((char*)mrc->sym, sizeof(char) * mrc->nsymbt);
        std::cout << "# reaching " << fin.tellg() << " bytes." << std::endl;
    }

    printf("# ncrs      : %d %d %d\n", mrc->ncrs[0], mrc->ncrs[1], mrc->ncrs[2]);
    printf("# mode      : %d\n", mrc->mode);
    printf("# ncrsstart : %d %d %d\n", mrc->ncrsstart[0], mrc->ncrsstart[1], mrc->ncrsstart[2]);
    printf("# mxyz      : %d %d %d\n", mrc->mxyz[0], mrc->mxyz[1], mrc->mxyz[2]);
    printf("# cella     : %.6f %.6f %.6f\n", mrc->cella[0], mrc->cella[1], mrc->cella[2]);
    printf("# cellb     : %.6f %.6f %.6f\n", mrc->cellb[0], mrc->cellb[1], mrc->cellb[2]);
    printf("# mapcrs    : %d %d %d\n", mrc->mapcrs[0], mrc->mapcrs[1], mrc->mapcrs[2]);
    printf("# dmin      : %.6f\n", mrc->dmin);
    printf("# dmax      : %.6f\n", mrc->dmax);
    printf("# dmean     : %.6f\n", mrc->dmean);
    printf("# ispg      : %d\n", mrc->ispg);
    printf("# nsymbt    : %d\n", mrc->nsymbt);
    printf("# origin    : %.6f %.6f %.6f\n", mrc->originxyz[0], mrc->originxyz[1], mrc->originxyz[2]);
    printf("# map       : %c%c%c%c.\n", mrc->map[0], mrc->map[1], mrc->map[2], mrc->map[3]);
    printf("# machst    : %#x %#x %#x %#x.\n", mrc->machst[0], mrc->machst[1], mrc->machst[2], mrc->machst[3]);
    printf("# rms       : %.6f\n", mrc->rms);
    printf("# nlabel    : %d\n", mrc->nlabel);

    if(mrc->mode != 2){
        std::cout << "# map line not stored in (double) type, can't handle this map." << std::endl;
        fin.close();
        exit(1);
    }

    if(!check_orthogonal(mrc->cellb)){
        std::cout << "# input grid is not orthogonal!!!" << std::endl;
        fin.close();
        exit(1);
    }
    /*
    if(!(mrc->ncrs[0] == mrc->ncrs[1] && mrc->ncrs[1] == mrc->ncrs[2])){
        std::cout << "# input grid is not cubic!!!" << std::endl;
    }
    */
    if(mrc->machst[0] == 0x44){
        std::cout << "# data stored in little-endian mode." << std::endl;
    }else if(mrc->machst[0] == 0x11){
        std::cout << "# data stored in big-endian mode, can't' handle this map." << std::endl;
        fin.close();
        exit(1);
    }else{
        printf("# machst    : %#x %#x %#x %#x, can't handle this map.\n", mrc->machst[0], mrc->machst[1], mrc->machst[2], mrc->machst[3]);
        fin.close();
        exit(1);
    }
    int* ncrs = mrc->ncrs;
    mrc->Nvoxel = ncrs[0] * ncrs[1] * ncrs[2];
    try{
      mrc->cube = new float**[ncrs[0]];
      for(int ic = 0; ic < ncrs[0]; ++ic){
        mrc->cube[ic] = new float*[ncrs[1]];
        for(int ir = 0; ir < ncrs[1]; ++ir){
          mrc->cube[ic][ir] = new float[ncrs[2]];
          for(int is = 0; is < ncrs[2]; ++is){
            mrc->cube[ic][ir][is] = 0.0f;
            fin.read((char*)&mrc->cube[ic][ir][is], sizeof(float) * 1);
          }
        }
      }
    }catch (const std::exception& err){
        std::cout << err.what() << std::endl;
        std::cout << "# Can't allocate map cube." << std::endl;
        fin.close();
        exit(1);
    }
    int order[3] = {mrc->mapcrs[0] - 1, mrc->mapcrs[1] - 1, mrc->mapcrs[2] - 1};

    for(int i = 0; i < 3; ++i){
        mrc->vsize[i] = mrc->cella[i] / mrc->mxyz[i];
        mrc->ORIGIN[i] = mrc->ncrsstart[order[i]] * mrc->vsize[i] + mrc->originxyz[i];
    }
    printf("# originnew : %.6f %.6f %.6f\n", mrc->ORIGIN[0], mrc->ORIGIN[1], mrc->ORIGIN[2]);
    fin.close();
    puts("# Done reading...");
}

void flatten(MRC* mrc){
  if(mrc->line)
    delete[] mrc->line;

  int* ncrs = mrc->ncrs;
  mrc->line = new float[mrc->Nvoxel];

  int i = 0, mode = get_mode(mrc->mapcrs), idx;
  for(int ic = 0; ic < ncrs[0]; ++ic){
    for(int ir = 0; ir < ncrs[1]; ++ir){
      for(int is = 0; is < ncrs[2]; ++is){
        idx = get_index(mode, i, ncrs[0], ncrs[1], ncrs[2]);
        mrc->line[idx] = mrc->cube[ic][ir][is];
        ++i;
      }
    }
  }
  //free mrc->cube
  if(mrc->cube == NULL)
    return;

  for(int ic = 0; ic < ncrs[0]; ++ic){
    for(int ir = 0; ir < ncrs[1]; ++ir){
      delete[] mrc->cube[ic][ir];
    }
    delete[] mrc->cube[ic];
  }
  delete[] mrc->cube;
  mrc->cube = NULL;
  printf("# Reassigned\n");
}

void to_cubic(MRC* mrc){
  //assert(mrc->mapcrs[0] == 1 && mrc->mapcrs[1] == 2 && mrc->mapcrs[2] == 3);
  if(mrc->cube){
    for(int ic = 0; ic < mrc->ncrs[0]; ++ic){
      for(int ir = 0; ir < mrc->ncrs[1]; ++ir)
        delete[] mrc->cube[ic][ir];
      delete[] mrc->cube[ic];
    }
    delete[] mrc->cube;
  }

  int sort[3], ncrs[3];
  for(int i = 0; i < 3; ++i){
    sort[i] = mrc->mapcrs[i] - 1;
    ncrs[i] = mrc->ncrs[sort[i]];
  }
  int mode = 1, i = 0, idx;

  mrc->cube = new float**[ncrs[0]];
  for(int ic = 0; ic < ncrs[0]; ++ic){
    mrc->cube[ic] = new float*[ncrs[1]];
    for(int ir = 0; ir < ncrs[1]; ++ir){
      mrc->cube[ic][ir] = new float[ncrs[2]];
      for(int is = 0; is < ncrs[2]; ++is){
        idx = get_index(1, i, ncrs[0], ncrs[1], ncrs[2]);
        mrc->cube[ic][ir][is] = mrc->line[idx];
        i += 1;
      }
    }
  }

  for(int i = 0; i < 3; ++i){
    mrc->mapcrs[i] = i + 1;
    mrc->ncrs[i]   = ncrs[i];
    mrc->mxyz[i]   = ncrs[i];
  }
}

void write_mrc(const char* file, MRC* mrc){
    std::ofstream fout(file, std::ios::out | std::ios::binary);
    if(fout.fail()){
        std::cout << "Can't write map to file : " << file << std::endl;
        exit(1);
    }
    fout.write((char*)mrc->ncrs,      sizeof(int) * 3);
    fout.write((char*)&mrc->mode,     sizeof(int));
    fout.write((char*)mrc->ncrsstart, sizeof(int) * 3);
    fout.write((char*)mrc->mxyz,      sizeof(int) * 3);
    fout.write((char*)mrc->cella,     sizeof(float) * 3);
    fout.write((char*)mrc->cellb,     sizeof(float) * 3);
    fout.write((char*)mrc->mapcrs,    sizeof(int) * 3);
    fout.write((char*)&mrc->dmin,     sizeof(float));
    fout.write((char*)&mrc->dmax,     sizeof(float));
    fout.write((char*)&mrc->dmean,    sizeof(float));
    fout.write((char*)&mrc->ispg,     sizeof(int));
    fout.write((char*)&mrc->nsymbt,   sizeof(int));
    //fout.seekp(25 * sizeof(int), std::ios::cur);
    fout.write((char*)mrc->extra,     sizeof(int) * 25);
    fout.write((char*)mrc->originxyz, sizeof(float) * 3);
    fout.write((char*)mrc->map,       sizeof(char) * 4);
    fout.write((char*)mrc->machst,    sizeof(char) * 4);
    fout.write((char*)&mrc->rms,      sizeof(float));
    fout.write((char*)&mrc->nlabel,   sizeof(int));
    fout.write((char*)mrc->label,     sizeof(char) * 10 * 80);
    assert(fout.tellp() == 1024);
    printf("# done writing header.\n");
  
    if(mrc->nsymbt > 0){
        fout.write((char*)mrc->sym,   sizeof(char) * mrc->nsymbt);
        printf("# done writing extended header.\n");
    }

    int mode = 1;
    int idx;
    for(int i = 0; i < mrc->Nvoxel; ++i){
        idx = get_index(mode, i, mrc->mxyz[0], mrc->mxyz[1], mrc->mxyz[2]);
        fout.write((char*)&mrc->line[idx], sizeof(float) * 1);
    }

    /* 
    int* ncrs = mrc->ncrs;
    for(int ic = 0; ic < ncrs[0]; ++ic){
      for(int ir = 0; ir < ncrs[1]; ++ir){
        for(int is = 0; is < ncrs[2]; ++is){
          fout.write((char*)&mrc->cube[ic][ir][is], sizeof(double) * 1);
        }
      }
    }
    */
    //printf("# done writing cube.\n");
    printf("# map written to %s.\n", file);
    fout.close();
}

