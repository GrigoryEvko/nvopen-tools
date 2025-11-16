// Function: sub_38B8110
// Address: 0x38b8110
//
__int64 __fastcall sub_38B8110(
        __int64 a1,
        __m128 a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  double v9; // xmm4_8
  double v10; // xmm5_8
  double v12; // xmm4_8
  double v13; // xmm5_8
  __int64 v14; // [rsp+8h] [rbp-18h] BYREF

  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_389FA00(a1, &v14, 1u)
    || (unsigned __int8)sub_38AB240(a1, v14, a2, *(double *)a3.m128i_i64, a4, a5, v9, v10, a8, a9) )
  {
    return 1;
  }
  else
  {
    return sub_38B7F60(a1, v14, a2, a3, a4, a5, v12, v13, a8, a9);
  }
}
