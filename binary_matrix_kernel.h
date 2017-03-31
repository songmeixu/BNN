//
// Created by songmeixu on 2017/2/16.
//

const static int binary_kernel_size = 32;

int ceiling(int x, int y) {
    assert(x>0);
    return 1 + ((x - 1) / y);
}

// 32 single float array ->  32 bits unsigned int
// low bit first
uint FloatVec2uint (const float* array) {
    uint rvalue=0;
    uint sign;

    for (int i = 0; i < 32; i++) {
        sign = (array[i]>=0);
        rvalue = rvalue | (sign<<i);
    }

    return rvalue;
}

// convert every 32 float in each row of '*in' into a uint,
// thus the out size is [in.NumRow(), ceil(in.NumCol()/32)]
int Mat2BMatByRow(const float *in, int in_row, int in_col, uint *out, int out_row, int out_col) {
    for (int r = 0; r < in_row; ++r) {
        for (int c = 0; c < in_col; c += 32) {
            out[r*out_col + c] = FloatVec2uint(in+r*in_col+c);
        }
    }
    return 0;
}

int MatBMat(const float *in1, int row1, int col1, const uint *in2, int row2, int col2, float *out) {
    assert (col1 <= row2 * binary_kernel_size && col1 > (row2-1) * binary_kernel_size);
    memset(out, 0, sizeof(float) * row1 * col2);

    for (int r = 0; r < row1; r++) {
        for (int c = 0; c < col2; c++) {
            int pos_out = r*col2+c;
            for (int k = 0; k < row2; ++k) {
                uint w_b = in2[k*col2+c];
                uint t = 1;
                int pos_in1 = r*col1 + k*binary_kernel_size;
                for (int b = 0; b < binary_kernel_size && (k*binary_kernel_size + b) < col1; ++b) {
                    if (w_b & (t << b))
                        out[pos_out] += in1[pos_in1 + b];
                    else
                        out[pos_out] -= in1[pos_in1 + b];
                }
            }
        }
    }
    return 0;
}

// Temporarily, we only support the number of neurons is 32x
// To support arbitrary neurons: pending
int BMatBMat(const uint *in1, int row1, int col1, const uint *in2, int row2, int col2, float *out) {
    assert (col1 == row2);
    memset(out, 0, sizeof(float) * row1 * col2);

    for (int r = 0; r < row1; r++) {
        for (int c = 0; c < col2; c++) {
            int target = r*col2+c;
            for (int k = 0; k < col1; ++k) {
                out[target] += __builtin_popcount(in1[r*row1+k] ^ in2[k*row2+c]);
            }
            out[target] = 32*col1 - (2*out[target]);
        }
    }
    return 0;
}
