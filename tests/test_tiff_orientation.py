import struct
import os
import numpy as np
from PIL import Image

def set_tiff_orientation_inplace(filepath, orientation_value):
    """
    Sets the Orientation tag (274) in a TIFF file without rewriting the image data.
    If the tag exists in the first IFD, it overwrites it.
    If it doesn't exist, it appends a new IFD to the end of the file and updates the header.
    orientation_value: 1-8 (1=Normal, 2=FlipH, 3=Rotate180, 4=FlipV, 5=Transpose, 6=Rotate90, 7=Transverse, 8=Rotate270)
    """
    with open(filepath, 'r+b') as f:
        f.seek(0)
        byte_order = f.read(2)
        if byte_order == b'II':
            endian = '<'
        elif byte_order == b'MM':
            endian = '>'
        else:
            raise ValueError("Not a valid TIFF")
        
        magic = struct.unpack(endian + 'H', f.read(2))[0]
        if magic != 42:
            raise ValueError("Not a valid TIFF")
        
        first_ifd_offset_pos = 4
        f.seek(first_ifd_offset_pos)
        offset = struct.unpack(endian + 'I', f.read(4))[0]
        
        # Parse the first IFD
        f.seek(offset)
        num_tags = struct.unpack(endian + 'H', f.read(2))[0]
        
        tags = []
        orientation_found = False
        
        for i in range(num_tags):
            tag_offset = offset + 2 + i * 12
            f.seek(tag_offset)
            tag_data = f.read(12)
            tag, fmt, count, value_or_offset = struct.unpack(endian + 'HHII', tag_data)
            
            if tag == 274: # Orientation
                # Format 3 is SHORT (2 bytes). The value fits in the 4-byte value/offset field.
                # Overwrite in place
                f.seek(tag_offset + 8)
                # Value is stored in the lower bytes or upper bytes depending on endianness?
                # Actually, struct.pack with 'I' for value_or_offset might be tricky for 'SHORT' since it only takes 2 bytes.
                # In II (little endian), the 2-byte short is the first 2 bytes, followed by 2 padding bytes.
                # Let's pack it correctly.
                # We'll just write the 2 bytes and leave the next 2 bytes alone (they should be 0).
                f.write(struct.pack(endian + 'H', orientation_value))
                orientation_found = True
            
            tags.append(tag_data)
            
        f.seek(offset + 2 + num_tags * 12)
        next_ifd_offset = struct.unpack(endian + 'I', f.read(4))[0]
        
        if orientation_found:
            return True
        
        # If not found, we append a new IFD to the end of the file.
        # Construct the new IFD
        new_tag_data = struct.pack(endian + 'HHII', 274, 3, 1, orientation_value)
        # In little-endian, a 2-byte value in a 4-byte field is the lower 2 bytes. The pack above works.
        # But wait, struct.pack with 'HHII' where the last I is the value. This correctly pads to 4 bytes!
        
        tags.append(new_tag_data)
        # Sort tags by tag ID (required by TIFF spec)
        def get_tag_id(data):
            return struct.unpack(endian + 'H', data[:2])[0]
        tags.sort(key=get_tag_id)
        
        f.seek(0, 2) # Go to end of file
        new_ifd_offset = f.tell()
        
        # Write new IFD
        f.write(struct.pack(endian + 'H', len(tags)))
        for tag_data in tags:
            f.write(tag_data)
        f.write(struct.pack(endian + 'I', next_ifd_offset))
        
        # Update the header to point to the new IFD
        f.seek(first_ifd_offset_pos)
        f.write(struct.pack(endian + 'I', new_ifd_offset))
        return True

def test_tiff_orientation():
    # 1. Create a small gradient image
    print("Creating a small gradient TIFF image...")
    width, height = 256, 256
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            gradient[y, x, 0] = x  # R: horizontal gradient
            gradient[y, x, 1] = y  # G: vertical gradient
            gradient[y, x, 2] = 128 # B: constant
            
    img = Image.fromarray(gradient)
    test_file = "tests/test_gradient.tif"
    # Save without orientation tag
    img.save(test_file, format="TIFF")
    
    # 2. Add orientation tag using our inplace method
    print("Setting orientation to 6 (Rotate90) via inplace IFD append...")
    set_tiff_orientation_inplace(test_file, 6)
    
    # 3. Read back with PIL to verify (PIL parses the orientation tag via ExifTags or info)
    img_read = Image.open(test_file)
    # Exif tags are usually in img.tag_v2
    orientation = img_read.tag_v2.get(274)
    print(f"Read back Orientation tag (274): {orientation}")
    
    # Let's also test changing it again (now it should overwrite in place)
    print("Changing orientation to 3 (Rotate180) via inplace overwrite...")
    set_tiff_orientation_inplace(test_file, 3)
    
    img_read2 = Image.open(test_file)
    orientation2 = img_read2.tag_v2.get(274)
    print(f"Read back Orientation tag (274): {orientation2}")
    
    assert orientation == 6, f"Expected 6, got {orientation}"
    assert orientation2 == 3, f"Expected 3, got {orientation2}"
    if os.path.exists(test_file):
        os.remove(test_file)
    print("Test passed successfully!")

if __name__ == "__main__":
    test_tiff_orientation()
