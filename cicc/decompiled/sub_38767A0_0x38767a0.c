// Function: sub_38767A0
// Address: 0x38767a0
//
__int64 ***__fastcall sub_38767A0(
        __int64 *a1,
        __int64 a2,
        __int64 **a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v12; // r15
  __int64 v15; // rsi
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  _QWORD v19[5]; // [rsp+8h] [rbp-28h] BYREF

  v12 = (__int64)(a1 + 33);
  a1[34] = *(_QWORD *)(a4 + 40);
  a1[35] = a4 + 24;
  v15 = *(_QWORD *)(a4 + 48);
  v19[0] = v15;
  if ( v15 )
  {
    sub_1623A60((__int64)v19, v15, 2);
    v16 = a1[33];
    if ( !v16 )
      goto LABEL_4;
  }
  else
  {
    v16 = a1[33];
    if ( !v16 )
      return sub_38761C0(a1, a2, a3, a5, a6, a7, a8, a9, a10, a11, a12);
  }
  sub_161E7C0(v12, v16);
LABEL_4:
  v17 = (unsigned __int8 *)v19[0];
  a1[33] = v19[0];
  if ( v17 )
    sub_1623210((__int64)v19, v17, v12);
  return sub_38761C0(a1, a2, a3, a5, a6, a7, a8, a9, a10, a11, a12);
}
