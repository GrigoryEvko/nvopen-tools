// Function: sub_38AA540
// Address: 0x38aa540
//
__int64 __fastcall sub_38AA540(
        __int64 a1,
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
  unsigned int v10; // r12d
  unsigned int v12; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v10 = sub_38AA270(a1, &v12, v13, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( !(_BYTE)v10 )
    sub_16267C0(a2, v12, v13[0]);
  return v10;
}
