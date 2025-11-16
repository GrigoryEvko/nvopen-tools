// Function: sub_191F410
// Address: 0x191f410
//
__int64 __fastcall sub_191F410(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
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
  __int64 v14; // rsi
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // rsi
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // rbx
  unsigned int v21; // r14d
  __int64 v22; // rsi
  __int64 v24; // [rsp+0h] [rbp-40h] BYREF
  __int64 v25; // [rsp+8h] [rbp-38h]
  __int64 v26; // [rsp+10h] [rbp-30h]

  sub_190DF50(a1, a2, a3, a4, a5, a6);
  v14 = *(_QWORD *)(a2 + 80);
  v24 = 0;
  v25 = 0;
  v26 = 0;
  if ( v14 )
    v14 -= 24;
  sub_191E690((__int64)&v24, v14);
  v15 = v25;
  v16 = v24;
  if ( v25 == v24 )
  {
    v21 = 0;
  }
  else
  {
    do
    {
      v17 = *(_QWORD *)(v15 - 8);
      v15 -= 8;
      sub_1918240(a1, v17);
    }
    while ( v16 != v15 );
    v16 = v25;
    v20 = v24;
    v21 = 0;
    if ( v24 != v25 )
    {
      do
      {
        v22 = *(_QWORD *)(v16 - 8);
        v16 -= 8;
        v21 |= sub_191E180(a1, v22, a7, a8, a9, a10, v18, v19, a13, a14);
      }
      while ( v20 != v16 );
      v16 = v24;
    }
  }
  if ( v16 )
    j_j___libc_free_0(v16, v26 - v16);
  return v21;
}
