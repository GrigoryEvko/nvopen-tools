// Function: sub_38761C0
// Address: 0x38761c0
//
__int64 ***__fastcall sub_38761C0(
        __int64 *a1,
        __int64 a2,
        __int64 **a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 ***result; // rax
  double v13; // xmm4_8
  double v14; // xmm5_8

  result = (__int64 ***)sub_3875200(a1, a2, *(double *)a4.m128_u64, a5, a6);
  if ( a3 )
    return sub_38744E0(a1, (__int64)result, a3, a4, a5, a6, a7, v13, v14, a10, a11);
  return result;
}
