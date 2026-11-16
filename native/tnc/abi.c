/* QHDOPT's packaging marker. The SciPy C source and header remain unmodified. */
#include "tnc.h"
#include <float.h>
#include <limits.h>

int qhdopt_tnc_abi(void) {
    return sizeof(int) == 4 && CHAR_BIT == 8 && sizeof(double) == 8 &&
           DBL_MANT_DIG == 53 && TNC_USERABORT == 7 ? 1 : 0;
}
