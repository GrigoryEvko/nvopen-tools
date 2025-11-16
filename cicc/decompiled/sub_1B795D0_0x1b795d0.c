// Function: sub_1B795D0
// Address: 0x1b795d0
//
__int64 __fastcall sub_1B795D0(
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
  double v11; // xmm4_8
  double v12; // xmm5_8
  __int64 v13; // r12
  __int64 v15; // [rsp+0h] [rbp-20h] BYREF
  char v16; // [rsp+8h] [rbp-18h]

  v10 = *a1;
  sub_1B76840((__int64)&v15, *a1, a2, *(double *)a3.m128_u64, a4, a5);
  if ( v16 )
    v13 = v15;
  else
    v13 = sub_1B785E0(v10, a2, a3, a4, a5, a6, v11, v12, a9, a10);
  sub_1B78FF0(v10, a3, a4, a5, a6, v11, v12, a9, a10);
  return v13;
}
