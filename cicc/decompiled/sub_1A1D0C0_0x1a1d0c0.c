// Function: sub_1A1D0C0
// Address: 0x1a1d0c0
//
_QWORD *__fastcall sub_1A1D0C0(__int64 *a1, __int64 a2, _BYTE *a3)
{
  bool v3; // zf
  _QWORD *v4; // r12
  __int64 v5; // r13
  unsigned __int64 *v6; // r14
  __m128i v7; // xmm0
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rsi
  unsigned __int8 *v12; // rsi
  __m128i v14; // [rsp+0h] [rbp-90h] BYREF
  __int64 v15; // [rsp+10h] [rbp-80h]
  __m128i v16; // [rsp+20h] [rbp-70h] BYREF
  __int64 v17; // [rsp+30h] [rbp-60h]
  __m128i v18; // [rsp+40h] [rbp-50h] BYREF
  __int16 v19; // [rsp+50h] [rbp-40h]

  v3 = *a3 == 0;
  LOWORD(v15) = 257;
  if ( !v3 )
  {
    v14.m128i_i64[0] = (__int64)a3;
    LOBYTE(v15) = 3;
  }
  v4 = sub_1648A60(64, 1u);
  if ( v4 )
    sub_15F9210((__int64)v4, *(_QWORD *)(*(_QWORD *)a2 + 24LL), a2, 0, 0, 0);
  v5 = a1[1];
  v6 = (unsigned __int64 *)a1[2];
  if ( (unsigned __int8)v15 > 1u )
  {
    v19 = 260;
    v18.m128i_i64[0] = (__int64)(a1 + 8);
    sub_14EC200(&v16, &v18, &v14);
  }
  else
  {
    v7 = _mm_loadu_si128(&v14);
    v17 = v15;
    v16 = v7;
  }
  if ( v5 )
  {
    sub_157E9D0(v5 + 40, (__int64)v4);
    v8 = *v6;
    v9 = v4[3];
    v4[4] = v6;
    v8 &= 0xFFFFFFFFFFFFFFF8LL;
    v4[3] = v8 | v9 & 7;
    *(_QWORD *)(v8 + 8) = v4 + 3;
    *v6 = *v6 & 7 | (unsigned __int64)(v4 + 3);
  }
  sub_164B780((__int64)v4, v16.m128i_i64);
  v10 = *a1;
  if ( *a1 )
  {
    v18.m128i_i64[0] = *a1;
    sub_1623A60((__int64)&v18, v10, 2);
    v11 = v4[6];
    if ( v11 )
      sub_161E7C0((__int64)(v4 + 6), v11);
    v12 = (unsigned __int8 *)v18.m128i_i64[0];
    v4[6] = v18.m128i_i64[0];
    if ( v12 )
      sub_1623210((__int64)&v18, v12, (__int64)(v4 + 6));
  }
  return v4;
}
