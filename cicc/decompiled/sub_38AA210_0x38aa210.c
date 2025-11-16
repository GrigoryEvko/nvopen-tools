// Function: sub_38AA210
// Address: 0x38aa210
//
__int64 __fastcall sub_38AA210(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 result; // rax
  double v11; // xmm4_8
  double v12; // xmm5_8

  if ( *(_DWORD *)(a1 + 64) == 376 )
    return sub_38A9970(a1, a2, 0, *(double *)a3.m128_u64, a4, a5);
  result = sub_388AF10(a1, 14, "expected '!' here");
  if ( !(_BYTE)result )
    return sub_38A2440(a1, a2, a3, a4, a5, a6, v11, v12, a9, a10);
  return result;
}
