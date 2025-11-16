// Function: sub_1AA6440
// Address: 0x1aa6440
//
__int64 __fastcall sub_1AA6440(
        __int64 a1,
        unsigned __int64 **a2,
        _QWORD *a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  unsigned __int64 *v13; // r13
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  double v16; // xmm4_8
  double v17; // xmm5_8
  __int64 result; // rax
  __int64 v19; // rsi
  __int64 v20; // r15
  __int64 v21; // rsi
  unsigned __int8 *v22; // rsi
  _QWORD v23[7]; // [rsp+8h] [rbp-38h] BYREF

  v13 = *a2;
  if ( !a3[6] )
  {
    if ( !v13 )
      BUG();
    v19 = v13[3];
    v20 = (__int64)(a3 + 6);
    v23[0] = v19;
    if ( v19 )
    {
      sub_1623A60((__int64)v23, v19, 2);
      v21 = a3[6];
      if ( v21 )
        sub_161E7C0(v20, v21);
      v22 = (unsigned __int8 *)v23[0];
      a3[6] = v23[0];
      if ( v22 )
        sub_1623210((__int64)v23, v22, v20);
      v13 = *a2;
    }
  }
  sub_157E9D0(a1, (__int64)a3);
  v14 = *v13;
  v15 = a3[3];
  a3[4] = v13;
  v14 &= 0xFFFFFFFFFFFFFFF8LL;
  a3[3] = v14 | v15 & 7;
  *(_QWORD *)(v14 + 8) = a3 + 3;
  *v13 = (unsigned __int64)(a3 + 3) | *v13 & 7;
  result = sub_1AA6390(a1, (__int64)a2, (__int64)a3, a4, a5, a6, a7, v16, v17, a10, a11);
  *a2 = a3 + 3;
  return result;
}
