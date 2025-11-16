// Function: sub_16B3820
// Address: 0x16b3820
//
__int64 __fastcall sub_16B3820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, float *a7)
{
  __int64 result; // rax
  float v8; // xmm0_4
  double v9; // [rsp+8h] [rbp-8h] BYREF

  result = sub_16B2400(a2, a5, a6, &v9);
  if ( !(_BYTE)result )
  {
    v8 = v9;
    *a7 = v8;
  }
  return result;
}
