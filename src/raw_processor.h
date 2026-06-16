#ifndef RAW_PROCESSOR_H
#define RAW_PROCESSOR_H

#include <string>
#include "libraw/libraw.h"

namespace RawProcessor {

// Loads a single RAW file and performs initial processing.
// Returns a pointer to a LibRaw object containing the image data.
// The caller is responsible for delete-ing the returned LibRaw pointer.
LibRaw* load_raw(const std::string& fn, bool debayer = true, bool half_size = false, int qual = 0, bool crop = true);

// Merges 4 pixel-shifted RAW objects (loaded via load_raw) into a single LibRaw object.
// Deletes proc[1], proc[2], and proc[3] during the merge.
// Returns a pointer to the merged LibRaw object (which is proc[0]).
LibRaw* merge_pixel_shift_raw(LibRaw *proc[4]);

} // namespace RawProcessor

#endif // RAW_PROCESSOR_H
