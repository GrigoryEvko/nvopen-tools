// Function: sub_1B79660
// Address: 0x1b79660
//
__int64 __fastcall sub_1B79660(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r13
  __int64 v11; // r12
  double v12; // xmm4_8
  double v13; // xmm5_8

  v10 = *a1;
  v11 = sub_1B75C50(*a1, a2, *(double *)a3.m128_u64, a4, a5);
  sub_1B78FF0(v10, a3, a4, a5, a6, v12, v13, a9, a10);
  return v11;
}
