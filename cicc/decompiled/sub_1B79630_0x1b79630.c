// Function: sub_1B79630
// Address: 0x1b79630
//
__int64 __fastcall sub_1B79630(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v14; // r12
  double v15; // xmm4_8
  double v16; // xmm5_8

  v14 = *a1;
  sub_1B78950(*a1, a2, a7, a8, a9, a10, a11, a12, a13, a14, a3, a4, a5, a6);
  return sub_1B78FF0(v14, a7, a8, a9, a10, v15, v16, a13, a14);
}
