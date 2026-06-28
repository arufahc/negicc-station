#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "image_capture.h"
#include "sony_camera_session.h"
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <fstream>

// Forward declarations


// Structure for Python's CapturedImage wrapper
typedef struct {
    PyObject_HEAD
    CapturedImage* cpp_img;
} PyCapturedImage;

// Structure for Python's CameraSession wrapper
typedef struct {
    PyObject_HEAD
    CameraSession* cpp_session;
} PyCameraSession;

static void PyCapturedImage_dealloc(PyCapturedImage* self) {
    if (self->cpp_img) {
        delete self->cpp_img;
        self->cpp_img = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// Getters for CapturedImage attributes
static PyObject* PyCapturedImage_get_shutter_speed(PyCapturedImage* self, void* closure) {
    if (!self->cpp_img) {
        PyErr_SetString(PyExc_RuntimeError, "CapturedImage C++ backend is null.");
        return nullptr;
    }
    return PyFloat_FromDouble(self->cpp_img->shutter_speed());
}

static PyObject* PyCapturedImage_get_iso(PyCapturedImage* self, void* closure) {
    if (!self->cpp_img) {
        PyErr_SetString(PyExc_RuntimeError, "CapturedImage C++ backend is null.");
        return nullptr;
    }
    return PyLong_FromLong(self->cpp_img->iso());
}

static PyObject* PyCapturedImage_get_capture_type(PyCapturedImage* self, void* closure) {
    if (!self->cpp_img) {
        PyErr_SetString(PyExc_RuntimeError, "CapturedImage C++ backend is null.");
        return nullptr;
    }
    return PyLong_FromLong(static_cast<long>(self->cpp_img->capture_type()));
}

static PyObject* PyCapturedImage_get_filepaths(PyCapturedImage* self, void* closure) {
    if (!self->cpp_img) {
        PyErr_SetString(PyExc_RuntimeError, "CapturedImage C++ backend is null.");
        return nullptr;
    }
    const auto& filepaths = self->cpp_img->filepaths();
    PyObject* py_list = PyList_New(filepaths.size());
    if (!py_list) return nullptr;

    for (size_t i = 0; i < filepaths.size(); ++i) {
        PyList_SetItem(py_list, i, PyUnicode_FromString(filepaths[i].c_str()));
    }
    return py_list;
}

static PyObject* PyCapturedImage_get_camera_model(PyCapturedImage* self, void* closure) {
    if (!self->cpp_img) {
        PyErr_SetString(PyExc_RuntimeError, "CapturedImage C++ backend is null.");
        return nullptr;
    }
    std::string model = self->cpp_img->camera_model();
    return PyUnicode_FromString(model.c_str());
}

// Methods on CapturedImage
static PyObject* PyCapturedImage_discard(PyCapturedImage* self, PyObject* Py_UNUSED(args)) {
    if (!self->cpp_img) {
        PyErr_SetString(PyExc_RuntimeError, "CapturedImage C++ backend is null.");
        return nullptr;
    }
    self->cpp_img->discard();
    Py_RETURN_NONE;
}

static PyObject* PyCapturedImage_to_numpy(PyCapturedImage* self, PyObject* args, PyObject* kwargs) {
    if (!self->cpp_img) {
        PyErr_SetString(PyExc_RuntimeError, "CapturedImage C++ backend is null.");
        return nullptr;
    }

    static const char* kwlist[] = {
        "half", "crosstalk_matrix", "it8_profile_path", "output_profile_path",
        "profile_film_base", "film_base", "exposure_comp", "g_gain", "b_gain", "pipeline",
        "it8_profile_bytes", "to_uint8", nullptr
    };
    int half = 0;
    PyObject* py_matrix = nullptr;
    const char* it8_profile_path = nullptr;
    const char* output_profile_path = nullptr;
    PyObject* py_profile_film_base = nullptr;
    PyObject* py_film_base = nullptr;
    float exposure_comp = 1.0f;
    float g_gain = 1.0f;
    float b_gain = 1.0f;
    const char* pipeline = "cuda";
    PyObject* py_icc_bytes = nullptr;
    int to_uint8 = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|pOzzOOfffzOp", const_cast<char**>(kwlist),
                                     &half, &py_matrix, &it8_profile_path, &output_profile_path,
                                     &py_profile_film_base, &py_film_base, &exposure_comp,
                                     &g_gain, &b_gain, &pipeline, &py_icc_bytes, &to_uint8)) {
        return nullptr;
    }

    std::vector<float> cc_matrix;
    if (py_matrix && py_matrix != Py_None) {
        if (!PyList_Check(py_matrix)) {
            PyErr_SetString(PyExc_TypeError, "crosstalk_matrix must be a list of 9 floats.");
            return nullptr;
        }
        Py_ssize_t len = PyList_Size(py_matrix);
        if (len != 9) {
            PyErr_SetString(PyExc_ValueError, "crosstalk_matrix must contain exactly 9 elements.");
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < 9; ++i) {
            PyObject* item = PyList_GetItem(py_matrix, i);
            if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "All elements of crosstalk_matrix must be floats.");
                return nullptr;
            }
            cc_matrix.push_back((float)PyFloat_AsDouble(item));
        }
    }

    std::vector<int> profile_film_base;
    if (py_profile_film_base && py_profile_film_base != Py_None) {
        if (!PyList_Check(py_profile_film_base)) {
            PyErr_SetString(PyExc_TypeError, "profile_film_base must be a list of 3 integers.");
            return nullptr;
        }
        if (PyList_Size(py_profile_film_base) != 3) {
            PyErr_SetString(PyExc_ValueError, "profile_film_base must contain exactly 3 elements.");
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < 3; ++i) {
            PyObject* item = PyList_GetItem(py_profile_film_base, i);
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "All elements of profile_film_base must be integers.");
                return nullptr;
            }
            profile_film_base.push_back((int)PyLong_AsLong(item));
        }
    }

    std::vector<int> film_base;
    if (py_film_base && py_film_base != Py_None) {
        if (!PyList_Check(py_film_base)) {
            PyErr_SetString(PyExc_TypeError, "film_base must be a list of 3 integers.");
            return nullptr;
        }
        if (PyList_Size(py_film_base) != 3) {
            PyErr_SetString(PyExc_ValueError, "film_base must contain exactly 3 elements.");
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < 3; ++i) {
            PyObject* item = PyList_GetItem(py_film_base, i);
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "All elements of film_base must be integers.");
                return nullptr;
            }
            film_base.push_back((int)PyLong_AsLong(item));
        }
    }

    // Resolve in-memory ICC bytes (preferred) or file path (legacy)
    const uint8_t* icc_data = nullptr;
    Py_ssize_t icc_data_size = 0;
    if (py_icc_bytes && py_icc_bytes != Py_None) {
        if (!PyBytes_Check(py_icc_bytes)) {
            PyErr_SetString(PyExc_TypeError, "it8_profile_bytes must be a bytes object.");
            return nullptr;
        }
        icc_data = reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(py_icc_bytes));
        icc_data_size = PyBytes_GET_SIZE(py_icc_bytes);
    }

    std::string it8_path = (it8_profile_path && !icc_data) ? it8_profile_path : "";
    std::string out_path = output_profile_path ? output_profile_path : "srgb";

    int w = 0, h = 0;
    std::string pipe_str = pipeline ? pipeline : "cuda";

    if (to_uint8) {
        if (half == 0) {
            PyErr_SetString(PyExc_ValueError, "uint8 preview pipeline only supports half_size=True (half=True).");
            return nullptr;
        }
        std::vector<uint8_t> buf;
        if (!self->cpp_img->get_preview_rgb8(w, h, buf, cc_matrix, it8_path, out_path,
                                             profile_film_base, film_base, exposure_comp, g_gain, b_gain,
                                             pipe_str, icc_data, (size_t)icc_data_size)) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to load/process RAW image buffer to preview rgb8.");
            return nullptr;
        }

        npy_intp dims[3] = { h, w, 3 };
        PyObject* arr = PyArray_SimpleNew(3, dims, NPY_UINT8);
        if (!arr) return nullptr;

        uint8_t* arr_data = static_cast<uint8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)));
        std::copy(buf.begin(), buf.end(), arr_data);

        return arr;
    } else {
        std::vector<uint16_t> buf;
        if (!self->cpp_img->get_linear_rgb(half != 0, w, h, buf, cc_matrix, it8_path, out_path,
                                           profile_film_base, film_base, exposure_comp, g_gain, b_gain,
                                           pipe_str, icc_data, (size_t)icc_data_size)) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to load/process RAW image buffer.");
            return nullptr;
        }

        npy_intp dims[3] = { h, w, 3 };
        PyObject* arr = PyArray_SimpleNew(3, dims, NPY_UINT16);
        if (!arr) return nullptr;

        uint16_t* arr_data = static_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)));
        std::copy(buf.begin(), buf.end(), arr_data);

        return arr;
    }
}

static PyObject* PyCapturedImage_write_tiff(PyCapturedImage* self, PyObject* args, PyObject* kwargs) {
    if (!self->cpp_img) {
        PyErr_SetString(PyExc_RuntimeError, "CapturedImage C++ backend is null.");
        return nullptr;
    }

    static const char* kwlist[] = {
        "output_path", "half", "crosstalk_matrix", "it8_profile_path", "output_profile_path",
        "profile_film_base", "film_base", "exposure_comp", "g_gain", "b_gain", "pipeline",
        "it8_profile_bytes", nullptr
    };
    const char* output_path = nullptr;
    int half = 0;
    PyObject* py_matrix = nullptr;
    const char* it8_profile_path = nullptr;
    const char* output_profile_path = nullptr;
    PyObject* py_profile_film_base = nullptr;
    PyObject* py_film_base = nullptr;
    float exposure_comp = 1.0f;
    float g_gain = 1.0f;
    float b_gain = 1.0f;
    const char* pipeline = "cuda";
    PyObject* py_icc_bytes = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|pOzzOOffzO", const_cast<char**>(kwlist),
                                     &output_path, &half, &py_matrix, &it8_profile_path, &output_profile_path,
                                     &py_profile_film_base, &py_film_base, &exposure_comp,
                                     &g_gain, &b_gain, &pipeline, &py_icc_bytes)) {
        return nullptr;
    }

    std::vector<float> cc_matrix;
    if (py_matrix && py_matrix != Py_None) {
        if (!PyList_Check(py_matrix)) {
            PyErr_SetString(PyExc_TypeError, "crosstalk_matrix must be a list of 9 floats.");
            return nullptr;
        }
        Py_ssize_t len = PyList_Size(py_matrix);
        if (len != 9) {
            PyErr_SetString(PyExc_ValueError, "crosstalk_matrix must contain exactly 9 elements.");
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < 9; ++i) {
            PyObject* item = PyList_GetItem(py_matrix, i);
            if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "All elements of crosstalk_matrix must be floats.");
                return nullptr;
            }
            cc_matrix.push_back((float)PyFloat_AsDouble(item));
        }
    }

    std::vector<int> profile_film_base;
    if (py_profile_film_base && py_profile_film_base != Py_None) {
        if (!PyList_Check(py_profile_film_base)) {
            PyErr_SetString(PyExc_TypeError, "profile_film_base must be a list of 3 integers.");
            return nullptr;
        }
        if (PyList_Size(py_profile_film_base) != 3) {
            PyErr_SetString(PyExc_ValueError, "profile_film_base must contain exactly 3 elements.");
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < 3; ++i) {
            PyObject* item = PyList_GetItem(py_profile_film_base, i);
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "All elements of profile_film_base must be integers.");
                return nullptr;
            }
            profile_film_base.push_back((int)PyLong_AsLong(item));
        }
    }

    std::vector<int> film_base;
    if (py_film_base && py_film_base != Py_None) {
        if (!PyList_Check(py_film_base)) {
            PyErr_SetString(PyExc_TypeError, "film_base must be a list of 3 integers.");
            return nullptr;
        }
        if (PyList_Size(py_film_base) != 3) {
            PyErr_SetString(PyExc_ValueError, "film_base must contain exactly 3 elements.");
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < 3; ++i) {
            PyObject* item = PyList_GetItem(py_film_base, i);
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "All elements of film_base must be integers.");
                return nullptr;
            }
            film_base.push_back((int)PyLong_AsLong(item));
        }
    }

    const uint8_t* icc_data = nullptr;
    Py_ssize_t icc_data_size = 0;
    if (py_icc_bytes && py_icc_bytes != Py_None) {
        if (!PyBytes_Check(py_icc_bytes)) {
            PyErr_SetString(PyExc_TypeError, "it8_profile_bytes must be a bytes object.");
            return nullptr;
        }
        icc_data = reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(py_icc_bytes));
        icc_data_size = PyBytes_GET_SIZE(py_icc_bytes);
    }

    std::string it8_path = (it8_profile_path && !icc_data) ? it8_profile_path : "";
    std::string out_path = output_profile_path ? output_profile_path : "srgb";

    std::string pipe_str = pipeline ? pipeline : "cuda";
    bool success = write_linear_tiff(*self->cpp_img, output_path, half != 0, cc_matrix,
                                     it8_path, out_path, profile_film_base, film_base,
                                     exposure_comp, g_gain, b_gain, pipe_str,
                                     icc_data, (size_t)icc_data_size);
    return PyBool_FromLong(success ? 1 : 0);
}

// Method table for CapturedImage type
static PyMethodDef PyCapturedImage_methods[] = {
    {"discard", (PyCFunction)PyCapturedImage_discard, METH_NOARGS, "Discard temporary RAW files from disk"},
    {"to_numpy", (PyCFunction)PyCapturedImage_to_numpy, METH_VARARGS | METH_KEYWORDS, "Convert captured image to linear RGB numpy array"},
    {"write_tiff", (PyCFunction)PyCapturedImage_write_tiff, METH_VARARGS | METH_KEYWORDS, "Write linear image directly to TIFF file"},
    {nullptr, nullptr, 0, nullptr}
};

// Getter table for CapturedImage attributes
static PyGetSetDef PyCapturedImage_getset[] = {
    {const_cast<char*>("shutter_speed"), (getter)PyCapturedImage_get_shutter_speed, nullptr, const_cast<char*>("Shutter speed in seconds"), nullptr},
    {const_cast<char*>("iso"), (getter)PyCapturedImage_get_iso, nullptr, const_cast<char*>("ISO value"), nullptr},
    {const_cast<char*>("capture_type"), (getter)PyCapturedImage_get_capture_type, nullptr, const_cast<char*>("Capture type (0=SINGLE, 1=SONY_PIXEL_SHIFT_4)"), nullptr},
    {const_cast<char*>("filepaths"), (getter)PyCapturedImage_get_filepaths, nullptr, const_cast<char*>("Temporary raw file paths"), nullptr},
    {const_cast<char*>("camera_model"), (getter)PyCapturedImage_get_camera_model, nullptr, const_cast<char*>("Camera model name"), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

// Constructor for CapturedImage to allow direct instantiation from Python
static int PyCapturedImage_init(PyCapturedImage* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"type", "shutter_speed", "iso", "filepaths", nullptr};
    int type_val = 0;
    double shutter_speed = 0.0;
    int iso = 100;
    PyObject* py_filepaths = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "idiO", const_cast<char**>(kwlist),
                                     &type_val, &shutter_speed, &iso, &py_filepaths)) {
        return -1;
    }

    if (!PyList_Check(py_filepaths)) {
        PyErr_SetString(PyExc_TypeError, "filepaths must be a list of strings.");
        return -1;
    }

    std::vector<std::string> filepaths;
    Py_ssize_t size = PyList_Size(py_filepaths);
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item = PyList_GetItem(py_filepaths, i);
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "All items in filepaths must be strings.");
            return -1;
        }
        filepaths.push_back(PyUnicode_AsUTF8(item));
    }

    ImageCaptureType type = static_cast<ImageCaptureType>(type_val);
    if (type != ImageCaptureType::SINGLE && type != ImageCaptureType::SONY_PIXEL_SHIFT_4) {
        PyErr_SetString(PyExc_ValueError, "Invalid capture type. Must be 0 (SINGLE) or 1 (SONY_PIXEL_SHIFT_4).");
        return -1;
    }

    self->cpp_img = new CapturedImage(type, shutter_speed, iso, filepaths);
    return 0;
}

// CapturedImage Type Specification
static PyTypeObject PyCapturedImage_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "negicc_station.CapturedImage",            /* tp_name */
    sizeof(PyCapturedImage),                   /* tp_basicsize */
    0,                                         /* tp_itemsize */
    (destructor)PyCapturedImage_dealloc,       /* tp_dealloc */
    0,                                         /* tp_vectorcall_offset */
    nullptr,                                   /* tp_getattr */
    nullptr,                                   /* tp_setattr */
    nullptr,                                   /* tp_as_async */
    nullptr,                                   /* tp_repr */
    nullptr,                                   /* tp_as_number */
    nullptr,                                   /* tp_as_sequence */
    nullptr,                                   /* tp_as_mapping */
    nullptr,                                   /* tp_hash */
    nullptr,                                   /* tp_call */
    nullptr,                                   /* tp_str */
    nullptr,                                   /* tp_getattro */
    nullptr,                                   /* tp_setattro */
    nullptr,                                   /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                        /* tp_flags */
    "Wrapper for C++ CapturedImage class",     /* tp_doc */
    nullptr,                                   /* tp_traverse */
    nullptr,                                   /* tp_clear */
    nullptr,                                   /* tp_richcompare */
    0,                                         /* tp_weaklistoffset */
    nullptr,                                   /* tp_iter */
    nullptr,                                   /* tp_iternext */
    PyCapturedImage_methods,                   /* tp_methods */
    nullptr,                                   /* tp_members */
    PyCapturedImage_getset,                    /* tp_getset */
    nullptr,                                   /* tp_base */
    nullptr,                                   /* tp_dict */
    nullptr,                                   /* tp_descr_get */
    nullptr,                                   /* tp_descr_set */
    0,                                         /* tp_dictoffset */
    (initproc)PyCapturedImage_init,            /* tp_init */
    nullptr,                                   /* tp_alloc */
    PyType_GenericNew,                         /* tp_new */
    nullptr,                                   /* tp_free */
    nullptr,                                   /* tp_is_gc */
    nullptr,                                   /* tp_bases */
    nullptr,                                   /* tp_mro */
    nullptr,                                   /* tp_cache */
    nullptr,                                   /* tp_subclasses */
    nullptr,                                   /* tp_weaklist */
    nullptr,                                   /* tp_del */
    0,                                         /* tp_version_tag */
    nullptr,                                   /* tp_finalize */
    nullptr,                                   /* tp_vectorcall */
};

static const uint32_t SUPPORTED_SHUTTER_SPEEDS[] = {
    0x12c000a, // 30s
    0xfa000a,  // 25s
    0xc8000a,  // 20s
    0x96000a,  // 15s
    0x82000a,  // 13s
    0x64000a,  // 10s
    0x50000a,  // 8s
    0x3c000a,  // 6s
    0x32000a,  // 5s
    0x28000a,  // 4s
    0x20000a,  // 3.2s
    0x19000a,  // 2.5s
    0x14000a,  // 2s
    0x10000a,  // 1.6s
    0xd000a,   // 1.3s
    0xa000a,   // 1s
    0x8000a,   // 0.8s
    0x6000a,   // 0.6s
    0x5000a,   // 0.5s
    0x4000a,   // 0.4s
    0x10003,   // 1/3s
    0x10004,   // 1/4s
    0x10005,   // 1/5s
    0x10006,   // 1/6s
    0x10008,   // 1/8s
    0x1000a,   // 1/10s
    0x1000d,   // 1/13s
    0x1000f,   // 1/15s
    0x10014,   // 1/20s
    0x10019,   // 1/25s
    0x1001e,   // 1/30s
    0x10028,   // 1/40s
    0x10032,   // 1/50s
    0x1003c,   // 1/60s
    0x10050,   // 1/80s
    0x10064,   // 1/100s
    0x1007d,   // 1/125s
    0x100a0,   // 1/160s
    0x100c8,   // 1/200s
    0x100fa,   // 1/250s
    0x10140,   // 1/320s
    0x10190,   // 1/400s
    0x101f4,   // 1/500s
    0x10280,   // 1/640s
    0x10320,   // 1/800s
    0x103e8,   // 1/1000s
    0x104e2,   // 1/1250s
    0x10640,   // 1/1600s
    0x107d0,   // 1/2000s
    0x109c4,   // 1/2500s
    0x10c80,   // 1/3200s
    0x10fa0,   // 1/4000s
    0x11388,   // 1/5000s
    0x11900,   // 1/6400s
    0x11f40,   // 1/8000s
};

static uint32_t map_shutter_speed(int numerator, int denominator) {
    if (denominator <= 0 || numerator <= 0) {
        return 0x10008; // Default to 1/8s on invalid input
    }
    double target = (double)numerator / (double)denominator;
    
    uint32_t best_val = 0x10008;
    double min_diff = -1.0;
    
    for (uint32_t val : SUPPORTED_SHUTTER_SPEEDS) {
        uint16_t num = val >> 16;
        uint16_t den = val & 0xFFFF;
        double speed = (den > 0) ? (double)num / (double)den : 0.0;
        double diff = std::abs(speed - target);
        if (min_diff < 0.0 || diff < min_diff) {
            min_diff = diff;
            best_val = val;
        }
    }
    return best_val;
}

// Module-level capture function
static PyObject* PyNegiccStation_capture(PyObject* Py_UNUSED(self), PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"type", "shutter_num", "shutter_den", nullptr};
    int type_int = 0;
    int shutter_num = 0;
    int shutter_den = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii", const_cast<char**>(kwlist), &type_int, &shutter_num, &shutter_den)) {
        return nullptr;
    }

    ImageCaptureType type = static_cast<ImageCaptureType>(type_int);
    if (type != ImageCaptureType::SINGLE && type != ImageCaptureType::SONY_PIXEL_SHIFT_4) {
        PyErr_SetString(PyExc_ValueError, "Invalid capture type. Must be 0 (SINGLE) or 1 (SONY_PIXEL_SHIFT_4).");
        return nullptr;
    }

    uint32_t shutter_val = map_shutter_speed(shutter_num, shutter_den);

    // Call actual C++ capture
    std::unique_ptr<CapturedImage> cpp_img = capture_image(type, shutter_val);
    if (!cpp_img) {
        PyErr_SetString(PyExc_RuntimeError, "Capture failed inside tethering session.");
        return nullptr;
    }

    // Allocate PyCapturedImage wrapper
    PyCapturedImage* py_img = (PyCapturedImage*)PyObject_New(PyCapturedImage, &PyCapturedImage_Type);
    if (!py_img) {
        return nullptr;
    }

    // Transfer ownership of unique_ptr to raw pointer inside Python object
    py_img->cpp_img = cpp_img.release();
    return (PyObject*)py_img;
}

// PyCameraSession implementation

static void PyCameraSession_dealloc(PyCameraSession* self) {
    if (self->cpp_session) {
        delete self->cpp_session;
        self->cpp_session = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int PyCameraSession_init(PyCameraSession* self, PyObject* Py_UNUSED(args), PyObject* Py_UNUSED(kwargs)) {
    self->cpp_session = new SonyCameraSession();
    return 0;
}

static PyObject* PyCameraSession_connect(PyCameraSession* self, PyObject* Py_UNUSED(args)) {
    if (!self->cpp_session) {
        PyErr_SetString(PyExc_RuntimeError, "CameraSession C++ backend is null.");
        return nullptr;
    }
    bool ok = self->cpp_session->initialize();
    if (!ok) {
        Py_RETURN_FALSE;
    }
    ok = self->cpp_session->configure_settings();
    if (!ok) {
        self->cpp_session->close();
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject* PyCameraSession_capture(PyCameraSession* self, PyObject* args, PyObject* kwargs) {
    if (!self->cpp_session) {
        PyErr_SetString(PyExc_RuntimeError, "CameraSession C++ backend is null.");
        return nullptr;
    }
    static const char* kwlist[] = {"type", "shutter_num", "shutter_den", nullptr};
    int type_int = 0;
    int shutter_num = 0;
    int shutter_den = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii", const_cast<char**>(kwlist), &type_int, &shutter_num, &shutter_den)) {
        return nullptr;
    }

    ImageCaptureType type = static_cast<ImageCaptureType>(type_int);
    if (type != ImageCaptureType::SINGLE && type != ImageCaptureType::SONY_PIXEL_SHIFT_4) {
        PyErr_SetString(PyExc_ValueError, "Invalid capture type. Must be 0 (SINGLE) or 1 (SONY_PIXEL_SHIFT_4).");
        return nullptr;
    }

    uint32_t shutter_val = map_shutter_speed(shutter_num, shutter_den);
    if (!self->cpp_session->set_shutter_speed(shutter_val)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set shutter speed on camera.");
        return nullptr;
    }

    CaptureOutput output;
    CaptureType capType = (type == ImageCaptureType::SONY_PIXEL_SHIFT_4)
                          ? CaptureType::SONY_PIXEL_SHIFT_4
                          : CaptureType::SINGLE;

    if (!self->cpp_session->capture(capType, output)) {
        PyErr_SetString(PyExc_RuntimeError, "Capture failed inside camera session.");
        return nullptr;
    }

    uint16_t numerator = shutter_val >> 16;
    uint16_t denominator = shutter_val & 0xFFFF;
    double shutterSec = (denominator > 0) ? (double)numerator / (double)denominator : 0.1;

    PyCapturedImage* py_img = (PyCapturedImage*)PyObject_New(PyCapturedImage, &PyCapturedImage_Type);
    if (!py_img) {
        return nullptr;
    }

    py_img->cpp_img = new CapturedImage(type, shutterSec, 100, output.filepaths);
    return (PyObject*)py_img;
}

static PyObject* PyCameraSession_close(PyCameraSession* self, PyObject* Py_UNUSED(args)) {
    if (self->cpp_session) {
        self->cpp_session->close();
    }
    Py_RETURN_NONE;
}

static PyMethodDef PyCameraSession_methods[] = {
    {"connect", (PyCFunction)PyCameraSession_connect, METH_NOARGS, "Connect to the camera and configure settings"},
    {"capture", (PyCFunction)PyCameraSession_capture, METH_VARARGS | METH_KEYWORDS, "Capture an image with the current connection"},
    {"close", (PyCFunction)PyCameraSession_close, METH_NOARGS, "Disconnect from the camera and release SDK"},
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject PyCameraSession_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "negicc_station.CameraSession",            /* tp_name */
    sizeof(PyCameraSession),                   /* tp_basicsize */
    0,                                         /* tp_itemsize */
    (destructor)PyCameraSession_dealloc,       /* tp_dealloc */
    0,                                         /* tp_vectorcall_offset */
    nullptr,                                   /* tp_getattr */
    nullptr,                                   /* tp_setattr */
    nullptr,                                   /* tp_as_async */
    nullptr,                                   /* tp_repr */
    nullptr,                                   /* tp_as_number */
    nullptr,                                   /* tp_as_sequence */
    nullptr,                                   /* tp_as_mapping */
    nullptr,                                   /* tp_hash */
    nullptr,                                   /* tp_call */
    nullptr,                                   /* tp_str */
    nullptr,                                   /* tp_getattro */
    nullptr,                                   /* tp_setattro */
    nullptr,                                   /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                        /* tp_flags */
    "Wrapper for CameraSession C++ class",     /* tp_doc */
    nullptr,                                   /* tp_traverse */
    nullptr,                                   /* tp_clear */
    nullptr,                                   /* tp_richcompare */
    0,                                         /* tp_weaklistoffset */
    nullptr,                                   /* tp_iter */
    nullptr,                                   /* tp_iternext */
    PyCameraSession_methods,                   /* tp_methods */
    nullptr,                                   /* tp_members */
    nullptr,                                   /* tp_getset */
    nullptr,                                   /* tp_base */
    nullptr,                                   /* tp_dict */
    nullptr,                                   /* tp_descr_get */
    nullptr,                                   /* tp_descr_set */
    0,                                         /* tp_dictoffset */
    (initproc)PyCameraSession_init,            /* tp_init */
    nullptr,                                   /* tp_alloc */
    PyType_GenericNew,                         /* tp_new */
    nullptr,                                   /* tp_free */
    nullptr,                                   /* tp_is_gc */
    nullptr,                                   /* tp_bases */
    nullptr,                                   /* tp_mro */
    nullptr,                                   /* tp_cache */
    nullptr,                                   /* tp_subclasses */
    nullptr,                                   /* tp_weaklist */
    nullptr,                                   /* tp_del */
    0,                                         /* tp_version_tag */
    nullptr,                                   /* tp_finalize */
    nullptr,                                   /* tp_vectorcall */
};

// Helper function to check if a Sony camera is connected via USB sysfs
static PyObject* PyNegiccStation_is_camera_connected(PyObject* /*self*/, PyObject* /*args*/) {
    bool connected = false;
    namespace fs = std::filesystem;
    try {
        if (fs::exists("/sys/bus/usb/devices")) {
            for (const auto& entry : fs::directory_iterator("/sys/bus/usb/devices")) {
                fs::path vendor_path = entry.path() / "idVendor";
                if (fs::exists(vendor_path)) {
                    std::ifstream file(vendor_path);
                    std::string vendor_id;
                    if (file >> vendor_id) {
                        // Sony Vendor ID is 054c (hexadecimal string in sysfs is usually lowercase 054c)
                        if (vendor_id == "054c" || vendor_id == "054C") {
                            connected = true;
                            break;
                        }
                    }
                }
            }
        }
    } catch (...) {
        // Ignore filesystem errors and return False
    }

    if (connected) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject* PyNegiccStation_cleanup_temp_files(PyObject* Py_UNUSED(self), PyObject* Py_UNUSED(args)) {
    cleanup_active_temp_files();
    Py_RETURN_NONE;
}

// Module method table
static PyMethodDef NegiccStation_module_methods[] = {
    {"capture", (PyCFunction)PyNegiccStation_capture, METH_VARARGS | METH_KEYWORDS, "Capture an image from tethered camera"},
    {"is_camera_connected", (PyCFunction)PyNegiccStation_is_camera_connected, METH_NOARGS, "Check if a Sony camera is connected via USB"},
    {"cleanup_temp_files", (PyCFunction)PyNegiccStation_cleanup_temp_files, METH_NOARGS, "Cleanup all active temporary RAW files"},
    {nullptr, nullptr, 0, nullptr}
};

// Module specification
static struct PyModuleDef negicc_station_module = {
    PyModuleDef_HEAD_INIT,
    "negicc_station",
    "Python bindings for tethered Sony capture and LibRaw decoding",
    -1,
    NegiccStation_module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

// Module initialization function
PyMODINIT_FUNC PyInit_negicc_station(void) {
    PyObject* m = PyModule_Create(&negicc_station_module);
    if (!m) return nullptr;

    // Initialize NumPy array C API
    import_array();

    if (PyType_Ready(&PyCapturedImage_Type) < 0) {
        Py_DECREF(m);
        return nullptr;
    }

    if (PyType_Ready(&PyCameraSession_Type) < 0) {
        Py_DECREF(m);
        return nullptr;
    }

    Py_INCREF(&PyCapturedImage_Type);
    if (PyModule_AddObject(m, "CapturedImage", (PyObject*)&PyCapturedImage_Type) < 0) {
        Py_DECREF(&PyCapturedImage_Type);
        Py_DECREF(m);
        return nullptr;
    }

    Py_INCREF(&PyCameraSession_Type);
    if (PyModule_AddObject(m, "CameraSession", (PyObject*)&PyCameraSession_Type) < 0) {
        Py_DECREF(&PyCameraSession_Type);
        Py_DECREF(&PyCapturedImage_Type);
        Py_DECREF(m);
        return nullptr;
    }

    // Constants
    PyModule_AddIntConstant(m, "CAPTURE_SINGLE", static_cast<int>(ImageCaptureType::SINGLE));
    PyModule_AddIntConstant(m, "CAPTURE_SONY_PIXEL_SHIFT_4", static_cast<int>(ImageCaptureType::SONY_PIXEL_SHIFT_4));

    return m;
}
