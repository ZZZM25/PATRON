#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <intrin.h>
#include <stdint.h>
#include <string.h>

#include "blst.h"
#include "longintrepr.h"


static const uint64_t FR_MODULUS_LIMBS[4] = {
    UINT64_C(0xffffffff00000001),
    UINT64_C(0x53bda402fffe5bfe),
    UINT64_C(0x3339d80809a1d805),
    UINT64_C(0x73eda753299d7d48),
};

static const uint64_t FR_R2_LIMBS[4] = {
    UINT64_C(0xc999e990f3f29c6d),
    UINT64_C(0x2b6cedcb87925c23),
    UINT64_C(0x05d314967254398f),
    UINT64_C(0x0748d9d99f59ff11),
};

static const uint64_t FR_MONT_INV = UINT64_C(0xfffffffeffffffff);


static int limbs_compare(const uint64_t left[4], const uint64_t right[4]) {
    for (int index = 3; index >= 0; --index) {
        if (left[index] < right[index]) {
            return -1;
        }
        if (left[index] > right[index]) {
            return 1;
        }
    }
    return 0;
}


static uint64_t multiply_accumulate(
    uint64_t accumulator,
    uint64_t left,
    uint64_t right,
    uint64_t *carry
) {
    uint64_t high;
    uint64_t low = _umul128(left, right, &high);
    uint64_t value;
    unsigned char carry0 = _addcarry_u64(0, low, accumulator, &value);
    unsigned char carry1 = _addcarry_u64(0, value, *carry, &value);
    *carry = high + (uint64_t)carry0 + (uint64_t)carry1;
    return value;
}


static void montgomery_multiply(
    uint64_t output[4],
    const uint64_t left[4],
    const uint64_t right[4]
) {
    uint64_t temporary[5] = {0, 0, 0, 0, 0};

    for (int outer = 0; outer < 4; ++outer) {
        uint64_t carry = 0;
        for (int inner = 0; inner < 4; ++inner) {
            temporary[inner] = multiply_accumulate(
                temporary[inner], left[inner], right[outer], &carry
            );
        }
        uint64_t top;
        unsigned char top_carry = _addcarry_u64(0, temporary[4], carry, &top);
        temporary[4] = top;

        uint64_t reduction = temporary[0] * FR_MONT_INV;
        carry = 0;
        for (int inner = 0; inner < 4; ++inner) {
            temporary[inner] = multiply_accumulate(
                temporary[inner], reduction, FR_MODULUS_LIMBS[inner], &carry
            );
        }
        unsigned char reduction_carry = _addcarry_u64(
            0, temporary[4], carry, &top
        );
        temporary[0] = temporary[1];
        temporary[1] = temporary[2];
        temporary[2] = temporary[3];
        temporary[3] = top;
        temporary[4] = (uint64_t)top_carry + (uint64_t)reduction_carry;
    }

    uint64_t reduced[5];
    unsigned char borrow = 0;
    for (int index = 0; index < 4; ++index) {
        borrow = _subborrow_u64(
            borrow, temporary[index], FR_MODULUS_LIMBS[index], &reduced[index]
        );
    }
    borrow = _subborrow_u64(borrow, temporary[4], 0, &reduced[4]);
    const uint64_t *selected = borrow == 0 ? reduced : temporary;
    memcpy(output, selected, 4 * sizeof(uint64_t));
}


static void field_add(
    uint64_t output[4],
    const uint64_t left[4],
    const uint64_t right[4]
) {
    uint64_t sum[4];
    unsigned char carry = 0;
    for (int index = 0; index < 4; ++index) {
        carry = _addcarry_u64(carry, left[index], right[index], &sum[index]);
    }

    uint64_t reduced[4];
    unsigned char borrow = 0;
    for (int index = 0; index < 4; ++index) {
        borrow = _subborrow_u64(
            borrow, sum[index], FR_MODULUS_LIMBS[index], &reduced[index]
        );
    }
    const uint64_t *selected = (carry != 0 || borrow == 0) ? reduced : sum;
    memcpy(output, selected, 4 * sizeof(uint64_t));
}


static void field_sub(
    uint64_t output[4],
    const uint64_t left[4],
    const uint64_t right[4]
) {
    uint64_t difference[4];
    unsigned char borrow = 0;
    for (int index = 0; index < 4; ++index) {
        borrow = _subborrow_u64(
            borrow, left[index], right[index], &difference[index]
        );
    }
    if (borrow != 0) {
        unsigned char carry = 0;
        for (int index = 0; index < 4; ++index) {
            carry = _addcarry_u64(
                carry, difference[index], FR_MODULUS_LIMBS[index],
                &difference[index]
            );
        }
    }
    memcpy(output, difference, 4 * sizeof(uint64_t));
}


static int python_long_to_field(PyObject *value, uint64_t output[4]) {
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "field coefficients must be integers");
        return -1;
    }

    PyLongObject *integer = (PyLongObject *)value;
    Py_ssize_t digit_count = Py_SIZE(integer);
    if (digit_count < 0) {
        PyErr_SetString(PyExc_ValueError, "field coefficients must not be negative");
        return -1;
    }

    output[0] = 0;
    output[1] = 0;
    output[2] = 0;
    output[3] = 0;
    for (Py_ssize_t index = 0; index < digit_count; ++index) {
        uint64_t digit_value = (uint64_t)integer->ob_digit[index];
        size_t bit_offset = (size_t)index * PyLong_SHIFT;
        size_t limb_index = bit_offset >> 6;
        unsigned int shift = (unsigned int)(bit_offset & 63);
        if (limb_index >= 4) {
            if (digit_value != 0) {
                PyErr_SetString(PyExc_ValueError, "field coefficient exceeds 256 bits");
                return -1;
            }
            continue;
        }
        output[limb_index] |= digit_value << shift;
        if (shift > 64 - PyLong_SHIFT && limb_index + 1 < 4) {
            output[limb_index + 1] |= digit_value >> (64 - shift);
        }
    }

    if (limbs_compare(output, FR_MODULUS_LIMBS) >= 0) {
        PyErr_SetString(PyExc_ValueError, "field coefficient is not canonical");
        return -1;
    }
    return 0;
}


static PyObject *field_to_python_long(const uint64_t value[4]) {
    return _PyLong_FromByteArray(
        (const unsigned char *)value,
        4 * sizeof(uint64_t),
        1,
        0
    );
}


static uint64_t *native_ntt_transform(
    PyObject *values,
    PyObject *stage_roots,
    PyObject *scale_value,
    Py_ssize_t *result_count
) {
    if (!PyList_Check(values) || !PyList_Check(stage_roots)) {
        PyErr_SetString(PyExc_TypeError, "values and stage_roots must be lists");
        return NULL;
    }
    Py_ssize_t count = PyList_GET_SIZE(values);
    if (count <= 0 || (count & (count - 1)) != 0) {
        PyErr_SetString(PyExc_ValueError, "NTT value count must be a positive power of two");
        return NULL;
    }
    Py_ssize_t stage_count = 0;
    for (Py_ssize_t size = count; size > 1; size >>= 1) {
        ++stage_count;
    }
    if (PyList_GET_SIZE(stage_roots) != stage_count) {
        PyErr_SetString(PyExc_ValueError, "stage root count does not match NTT size");
        return NULL;
    }
    if ((size_t)count > SIZE_MAX / (4 * sizeof(uint64_t)) ||
        (size_t)(stage_count > 0 ? stage_count : 1) > SIZE_MAX / (4 * sizeof(uint64_t))) {
        return PyErr_NoMemory(), NULL;
    }

    uint64_t *field_values = PyMem_Malloc(
        (size_t)count * 4 * sizeof(uint64_t)
    );
    uint64_t *roots = PyMem_Malloc(
        (size_t)(stage_count > 0 ? stage_count : 1) * 4 * sizeof(uint64_t)
    );
    if (field_values == NULL || roots == NULL) {
        PyMem_Free(field_values);
        PyMem_Free(roots);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t index = 0; index < count; ++index) {
        uint64_t standard[4];
        if (python_long_to_field(PyList_GET_ITEM(values, index), standard) < 0) {
            PyMem_Free(field_values);
            PyMem_Free(roots);
            return NULL;
        }
        montgomery_multiply(field_values + index * 4, standard, FR_R2_LIMBS);
    }
    for (Py_ssize_t index = 0; index < stage_count; ++index) {
        uint64_t standard[4];
        if (python_long_to_field(PyList_GET_ITEM(stage_roots, index), standard) < 0) {
            PyMem_Free(field_values);
            PyMem_Free(roots);
            return NULL;
        }
        montgomery_multiply(roots + index * 4, standard, FR_R2_LIMBS);
    }
    uint64_t standard_scale[4];
    uint64_t montgomery_scale[4];
    uint64_t standard_one[4] = {1, 0, 0, 0};
    uint64_t montgomery_one[4];
    if (python_long_to_field(scale_value, standard_scale) < 0) {
        PyMem_Free(field_values);
        PyMem_Free(roots);
        return NULL;
    }
    montgomery_multiply(montgomery_scale, standard_scale, FR_R2_LIMBS);
    montgomery_multiply(montgomery_one, standard_one, FR_R2_LIMBS);

    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t reversed_index = 0;
    for (Py_ssize_t current_index = 1; current_index < count; ++current_index) {
        Py_ssize_t bit = count >> 1;
        while (reversed_index & bit) {
            reversed_index ^= bit;
            bit >>= 1;
        }
        reversed_index ^= bit;
        if (current_index < reversed_index) {
            for (int limb_index = 0; limb_index < 4; ++limb_index) {
                uint64_t temporary = field_values[current_index * 4 + limb_index];
                field_values[current_index * 4 + limb_index] =
                    field_values[reversed_index * 4 + limb_index];
                field_values[reversed_index * 4 + limb_index] = temporary;
            }
        }
    }

    Py_ssize_t current_length = 2;
    Py_ssize_t stage_index = 0;
    while (current_length <= count) {
        Py_ssize_t half_length = current_length >> 1;
        const uint64_t *step_root = roots + stage_index * 4;
        for (Py_ssize_t block_start = 0; block_start < count; block_start += current_length) {
            uint64_t current_root[4];
            memcpy(current_root, montgomery_one, sizeof(current_root));
            for (Py_ssize_t offset = 0; offset < half_length; ++offset) {
                uint64_t left[4];
                uint64_t right[4];
                uint64_t next_root[4];
                uint64_t *left_slot = field_values + (block_start + offset) * 4;
                uint64_t *right_slot = field_values +
                    (block_start + offset + half_length) * 4;
                memcpy(left, left_slot, sizeof(left));
                montgomery_multiply(right, right_slot, current_root);
                field_add(left_slot, left, right);
                field_sub(right_slot, left, right);
                montgomery_multiply(next_root, current_root, step_root);
                memcpy(current_root, next_root, sizeof(current_root));
            }
        }
        ++stage_index;
        current_length <<= 1;
    }
    for (Py_ssize_t index = 0; index < count; ++index) {
        uint64_t scaled[4];
        montgomery_multiply(
            scaled, field_values + index * 4, montgomery_scale
        );
        memcpy(field_values + index * 4, scaled, sizeof(scaled));
    }
    Py_END_ALLOW_THREADS

    PyMem_Free(roots);
    *result_count = count;
    return field_values;
}


static PyObject *native_ntt(PyObject *self, PyObject *args) {
    PyObject *values;
    PyObject *stage_roots;
    PyObject *scale;
    if (!PyArg_ParseTuple(args, "OOO:ntt", &values, &stage_roots, &scale)) {
        return NULL;
    }
    Py_ssize_t count;
    uint64_t *field_values = native_ntt_transform(
        values, stage_roots, scale, &count
    );
    if (field_values == NULL) {
        return NULL;
    }
    PyObject *result = PyList_New(count);
    if (result == NULL) {
        PyMem_Free(field_values);
        return NULL;
    }
    uint64_t standard_one[4] = {1, 0, 0, 0};
    for (Py_ssize_t index = 0; index < count; ++index) {
        uint64_t standard[4];
        montgomery_multiply(
            standard, field_values + index * 4, standard_one
        );
        PyObject *python_value = field_to_python_long(standard);
        if (python_value == NULL) {
            Py_DECREF(result);
            PyMem_Free(field_values);
            return NULL;
        }
        PyList_SET_ITEM(result, index, python_value);
    }
    PyMem_Free(field_values);
    return result;
}


static PyObject *native_inverse_ntt_packed(PyObject *self, PyObject *args) {
    PyObject *values;
    PyObject *stage_roots;
    PyObject *scale;
    if (!PyArg_ParseTuple(
        args, "OOO:inverse_ntt_packed", &values, &stage_roots, &scale
    )) {
        return NULL;
    }
    Py_ssize_t count;
    uint64_t *field_values = native_ntt_transform(
        values, stage_roots, scale, &count
    );
    if (field_values == NULL) {
        return NULL;
    }
    if ((size_t)count > (SIZE_MAX - 4) / 32) {
        PyMem_Free(field_values);
        return PyErr_NoMemory();
    }
    Py_ssize_t output_length = 4 + count * 32;
    PyObject *result = PyBytes_FromStringAndSize(NULL, output_length);
    if (result == NULL) {
        PyMem_Free(field_values);
        return NULL;
    }
    unsigned char *output = (unsigned char *)PyBytes_AS_STRING(result);
    uint32_t encoded_count = (uint32_t)count;
    output[0] = (unsigned char)(encoded_count >> 24);
    output[1] = (unsigned char)(encoded_count >> 16);
    output[2] = (unsigned char)(encoded_count >> 8);
    output[3] = (unsigned char)encoded_count;
    uint64_t standard_one[4] = {1, 0, 0, 0};
    for (Py_ssize_t index = 0; index < count; ++index) {
        uint64_t standard[4];
        montgomery_multiply(
            standard, field_values + index * 4, standard_one
        );
        unsigned char *encoded = output + 4 + index * 32;
        for (int limb_index = 0; limb_index < 4; ++limb_index) {
            uint64_t limb = standard[3 - limb_index];
            for (int byte_index = 0; byte_index < 8; ++byte_index) {
                encoded[limb_index * 8 + byte_index] =
                    (unsigned char)(limb >> (56 - byte_index * 8));
            }
        }
    }
    PyMem_Free(field_values);
    return result;
}


typedef struct {
    Py_ssize_t count;
    blst_p1_affine *points;
} PreparedG1Points;


static const char *PREPARED_G1_CAPSULE_NAME = "kzg.native.prepared_g1";


static void prepared_g1_capsule_destructor(PyObject *capsule) {
    PreparedG1Points *prepared = PyCapsule_GetPointer(
        capsule, PREPARED_G1_CAPSULE_NAME
    );
    if (prepared == NULL) {
        PyErr_Clear();
        return;
    }
    PyMem_Free(prepared->points);
    PyMem_Free(prepared);
}


static PyObject *native_prepare_g1_points(PyObject *self, PyObject *args) {
    const char *encoded_points;
    Py_ssize_t encoded_length;
    if (!PyArg_ParseTuple(
        args, "y#:prepare_g1_points", &encoded_points, &encoded_length
    )) {
        return NULL;
    }
    if (encoded_length <= 0 || encoded_length % 48 != 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "compressed G1 points must contain a positive multiple of 48 bytes"
        );
        return NULL;
    }
    Py_ssize_t point_count = encoded_length / 48;
    if ((size_t)point_count > SIZE_MAX / sizeof(blst_p1_affine)) {
        return PyErr_NoMemory();
    }
    PreparedG1Points *prepared = PyMem_Malloc(sizeof(PreparedG1Points));
    if (prepared == NULL) {
        return PyErr_NoMemory();
    }
    prepared->count = point_count;
    prepared->points = PyMem_Malloc(
        (size_t)point_count * sizeof(blst_p1_affine)
    );
    if (prepared->points == NULL) {
        PyMem_Free(prepared);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t index = 0; index < point_count; ++index) {
        BLST_ERROR error = blst_p1_uncompress(
            &prepared->points[index],
            (const byte *)encoded_points + (size_t)index * 48
        );
        if (error != BLST_SUCCESS ||
            !blst_p1_affine_in_g1(&prepared->points[index])) {
            PyMem_Free(prepared->points);
            PyMem_Free(prepared);
            PyErr_Format(PyExc_ValueError, "invalid G1 point at index %zd", index);
            return NULL;
        }
    }

    PyObject *capsule = PyCapsule_New(
        prepared, PREPARED_G1_CAPSULE_NAME, prepared_g1_capsule_destructor
    );
    if (capsule == NULL) {
        PyMem_Free(prepared->points);
        PyMem_Free(prepared);
    }
    return capsule;
}


static PyObject *native_msm_g1_prepared(PyObject *self, PyObject *args) {
    PyObject *capsule;
    PyObject *scalars;
    if (!PyArg_ParseTuple(
        args, "OO:msm_g1_prepared", &capsule, &scalars
    )) {
        return NULL;
    }
    PreparedG1Points *prepared = PyCapsule_GetPointer(
        capsule, PREPARED_G1_CAPSULE_NAME
    );
    if (prepared == NULL) {
        return NULL;
    }
    if (!PyList_Check(scalars)) {
        PyErr_SetString(PyExc_TypeError, "scalars must be a list");
        return NULL;
    }
    Py_ssize_t scalar_count = PyList_GET_SIZE(scalars);
    if (scalar_count <= 0 || scalar_count > prepared->count) {
        PyErr_SetString(
            PyExc_ValueError,
            "scalar count must be positive and no greater than prepared point count"
        );
        return NULL;
    }

    size_t point_count = (size_t)scalar_count;
    if (point_count > SIZE_MAX / 32) {
        return PyErr_NoMemory();
    }
    byte *scalar_bytes = PyMem_Calloc(point_count, 32);
    if (scalar_bytes == NULL) {
        return PyErr_NoMemory();
    }
    for (Py_ssize_t index = 0; index < scalar_count; ++index) {
        uint64_t limbs[4];
        if (python_long_to_field(PyList_GET_ITEM(scalars, index), limbs) < 0) {
            PyMem_Free(scalar_bytes);
            return NULL;
        }
        for (int limb_index = 0; limb_index < 4; ++limb_index) {
            uint64_t limb = limbs[limb_index];
            for (int byte_index = 0; byte_index < 8; ++byte_index) {
                scalar_bytes[(size_t)index * 32 + (size_t)limb_index * 8 + byte_index] =
                    (byte)(limb >> (byte_index * 8));
            }
        }
    }

    size_t scratch_size = blst_p1s_mult_pippenger_scratch_sizeof(point_count);
    limb_t *scratch = PyMem_Malloc(scratch_size);
    if (scratch == NULL) {
        PyMem_Free(scalar_bytes);
        return PyErr_NoMemory();
    }

    const blst_p1_affine *point_pointers[2] = {prepared->points, NULL};
    const byte *scalar_pointers[2] = {scalar_bytes, NULL};
    blst_p1 result;
    byte compressed[48];
    Py_BEGIN_ALLOW_THREADS
    blst_p1s_mult_pippenger(
        &result,
        point_pointers,
        point_count,
        scalar_pointers,
        255,
        scratch
    );
    blst_p1_compress(compressed, &result);
    Py_END_ALLOW_THREADS

    PyMem_Free(scratch);
    PyMem_Free(scalar_bytes);
    return PyBytes_FromStringAndSize((const char *)compressed, sizeof(compressed));
}


static PyObject *native_aggregate_polynomials(PyObject *self, PyObject *args) {
    PyObject *polynomials;
    PyObject *weights;
    if (!PyArg_ParseTuple(args, "OO:aggregate_polynomials", &polynomials, &weights)) {
        return NULL;
    }
    if (!PyList_Check(polynomials) || !PyList_Check(weights)) {
        PyErr_SetString(PyExc_TypeError, "polynomials and weights must be lists");
        return NULL;
    }

    Py_ssize_t polynomial_count = PyList_GET_SIZE(polynomials);
    if (polynomial_count <= 0) {
        PyErr_SetString(PyExc_ValueError, "at least one polynomial is required");
        return NULL;
    }
    if (PyList_GET_SIZE(weights) != polynomial_count) {
        PyErr_SetString(
            PyExc_ValueError,
            "polynomials and weights must have the same length"
        );
        return NULL;
    }

    Py_ssize_t maximum_length = 0;
    for (Py_ssize_t polynomial_index = 0; polynomial_index < polynomial_count; ++polynomial_index) {
        PyObject *polynomial = PyList_GET_ITEM(polynomials, polynomial_index);
        if (!PyList_Check(polynomial)) {
            PyErr_SetString(PyExc_TypeError, "every polynomial must be a list");
            return NULL;
        }
        Py_ssize_t length = PyList_GET_SIZE(polynomial);
        if (length > maximum_length) {
            maximum_length = length;
        }
    }
    if (maximum_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "polynomials cannot be empty");
        return NULL;
    }

    uint64_t *montgomery_weights = PyMem_Malloc(
        (size_t)polynomial_count * 4 * sizeof(uint64_t)
    );
    if (montgomery_weights == NULL) {
        return PyErr_NoMemory();
    }
    for (Py_ssize_t polynomial_index = 0; polynomial_index < polynomial_count; ++polynomial_index) {
        uint64_t weight[4];
        if (python_long_to_field(PyList_GET_ITEM(weights, polynomial_index), weight) < 0) {
            PyMem_Free(montgomery_weights);
            return NULL;
        }
        montgomery_multiply(
            montgomery_weights + polynomial_index * 4,
            weight,
            FR_R2_LIMBS
        );
    }

    PyObject *result = PyList_New(maximum_length);
    if (result == NULL) {
        PyMem_Free(montgomery_weights);
        return NULL;
    }

    for (Py_ssize_t coefficient_index = 0; coefficient_index < maximum_length; ++coefficient_index) {
        uint64_t accumulator[4] = {0, 0, 0, 0};
        for (Py_ssize_t polynomial_index = 0; polynomial_index < polynomial_count; ++polynomial_index) {
            PyObject *polynomial = PyList_GET_ITEM(polynomials, polynomial_index);
            if (coefficient_index >= PyList_GET_SIZE(polynomial)) {
                continue;
            }
            uint64_t coefficient[4];
            uint64_t product[4];
            if (python_long_to_field(
                PyList_GET_ITEM(polynomial, coefficient_index), coefficient
            ) < 0) {
                Py_DECREF(result);
                PyMem_Free(montgomery_weights);
                return NULL;
            }
            montgomery_multiply(
                product,
                coefficient,
                montgomery_weights + polynomial_index * 4
            );
            field_add(accumulator, accumulator, product);
        }
        PyObject *python_value = field_to_python_long(accumulator);
        if (python_value == NULL) {
            Py_DECREF(result);
            PyMem_Free(montgomery_weights);
            return NULL;
        }
        PyList_SET_ITEM(result, coefficient_index, python_value);
    }

    PyMem_Free(montgomery_weights);
    return result;
}


static PyMethodDef native_methods[] = {
    {
        "ntt",
        native_ntt,
        METH_VARARGS,
        "Run an NTT using native BLS12-381 scalar-field arithmetic."
    },
    {
        "inverse_ntt_packed",
        native_inverse_ntt_packed,
        METH_VARARGS,
        "Run an inverse NTT and return DailyStore packed coefficients."
    },
    {
        "prepare_g1_points",
        native_prepare_g1_points,
        METH_VARARGS,
        "Decode and validate compressed G1 points for repeated native MSM."
    },
    {
        "msm_g1_prepared",
        native_msm_g1_prepared,
        METH_VARARGS,
        "Run BLST Pippenger MSM over prepared G1 points."
    },
    {
        "aggregate_polynomials",
        native_aggregate_polynomials,
        METH_VARARGS,
        "Aggregate BLS12-381 scalar-field polynomials using native arithmetic."
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef native_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native KZG hot paths.",
    -1,
    native_methods,
};


PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&native_module);
}
