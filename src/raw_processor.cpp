#include "raw_processor.h"
#include <iostream>
#include <cstdio>
#include <algorithm>

namespace RawProcessor {

LibRaw* load_raw(const std::string& fn, bool debayer, bool half_size, int qual, bool crop) {
  int ret;
  LibRaw* proc = new LibRaw();

  printf("Loading RAW file %s\n", fn.c_str());
  if ((ret = proc->open_file(fn.c_str())) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot open %s: %s\n", fn.c_str(), libraw_strerror(ret));
    delete proc;
    return nullptr;
  }
  printf("Image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  if (debayer)
    printf("Debayer quality: %d\n", qual);

  if ((ret = proc->unpack()) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot unpack %s: %s\n", fn.c_str(), libraw_strerror(ret));
    delete proc;
    return nullptr;
  }
  if (!(proc->imgdata.idata.filters || proc->imgdata.idata.colors == 1)) {
    printf("Only Bayer-pattern RAW files supported, sorry....\n");
    delete proc;
    return nullptr;
  }

  proc->imgdata.params.output_bps = 16;
  proc->imgdata.params.user_flip = 0;
  proc->imgdata.params.gamm[0] = 1;
  proc->imgdata.params.gamm[1] = 1;
  proc->imgdata.params.no_auto_bright = 1;
  proc->imgdata.params.no_auto_scale = 1;
  proc->imgdata.params.highlight = 1;
  proc->imgdata.params.output_color = 0;
  proc->imgdata.params.output_tiff = 1;
  if (!debayer) {
    proc->imgdata.params.no_interpolation = 1;
    proc->raw2image();
    proc->subtract_black();
  } else {
    proc->imgdata.params.half_size = half_size;
    proc->imgdata.params.user_qual = qual;
    proc->imgdata.params.use_auto_wb = 0;
    proc->imgdata.params.user_mul[0] = 1;
    proc->imgdata.params.user_mul[1] = 1;
    proc->imgdata.params.user_mul[2] = 1;
    proc->imgdata.params.user_mul[3] = 1;
    if (crop &&
        (proc->imgdata.sizes.raw_inset_crop.cleft || proc->imgdata.sizes.raw_inset_crop.ctop)) {
      proc->imgdata.params.cropbox[0] = proc->imgdata.sizes.raw_inset_crop.cleft;
      proc->imgdata.params.cropbox[1] = proc->imgdata.sizes.raw_inset_crop.ctop;
      proc->imgdata.params.cropbox[2] = proc->imgdata.sizes.raw_inset_crop.cwidth;
      proc->imgdata.params.cropbox[3] = proc->imgdata.sizes.raw_inset_crop.cheight;
    }
    proc->dcraw_process();
  }
  printf("Processed image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  return proc;
}

LibRaw* merge_pixel_shift_raw(LibRaw *proc[4]) {
  printf("Merging 4 images...\n");

  int movements[4][2] = {
    // x, y movements
    {0, 0},
    {0, 1},
    {-1, 1},
    {-1, 0},
  };

#define P(n, r, c) proc[n]->imgdata.image[(r) * proc[n]->imgdata.sizes.iwidth + (c)]
#define FOR_PIXEL for (int r = 0; r < proc[mi]->imgdata.sizes.iheight - 1; ++r) \
    for (int c = 1; c < proc[mi]->imgdata.sizes.iwidth; ++c)

  for (int mi = 0; mi < 2; ++mi) {
    int dc = movements[mi][0];
    int dr = movements[mi][1];

    FOR_PIXEL {
      int col = proc[mi]->COLOR(r, c);
      if (col & 1)
        P(0, r+dr, c+dc)[1] = P(mi, r, c)[col];
      else
        P(0, r+dr, c+dc)[col] = P(mi, r, c)[col];
    }
    if (mi > 0) {
      proc[mi]->recycle();
      delete proc[mi];
    }
  }

  for (int mi = 2; mi < 4; ++mi) {
    int dc = movements[mi][0];
    int dr = movements[mi][1];

    FOR_PIXEL {
      int col = proc[mi]->COLOR(r, c);
      if (col & 1)
        P(0, r+dr, c+dc)[1] = (P(mi, r, c)[col] + P(0, r+dr, c+dc)[1]) / 2;
      else
        P(0, r+dr, c+dc)[col] = P(mi, r, c)[col];
    }
    proc[mi]->recycle();
    delete proc[mi];
  }
  proc[0]->imgdata.idata.colors = 3;
  return proc[0];
}

} // namespace RawProcessor
