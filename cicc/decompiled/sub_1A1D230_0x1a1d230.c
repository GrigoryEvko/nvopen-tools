// Function: sub_1A1D230
// Address: 0x1a1d230
//
_QWORD *__fastcall sub_1A1D230(__int64 *a1, __int64 a2, unsigned __int8 a3, const __m128i *a4)
{
  _QWORD *v6; // r12
  __int64 v7; // r14
  unsigned __int64 *v8; // r15
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rsi
  unsigned __int8 *v13; // rsi
  __m128i v15; // [rsp+10h] [rbp-70h] BYREF
  __int64 v16; // [rsp+20h] [rbp-60h]
  __m128i v17; // [rsp+30h] [rbp-50h] BYREF
  __int16 v18; // [rsp+40h] [rbp-40h]

  v6 = sub_1648A60(64, 1u);
  if ( v6 )
    sub_15F9210((__int64)v6, *(_QWORD *)(*(_QWORD *)a2 + 24LL), a2, 0, a3, 0);
  v7 = a1[1];
  v8 = (unsigned __int64 *)a1[2];
  if ( a4[1].m128i_i8[0] > 1u )
  {
    v18 = 260;
    v17.m128i_i64[0] = (__int64)(a1 + 8);
    sub_14EC200(&v15, &v17, a4);
  }
  else
  {
    v16 = a4[1].m128i_i64[0];
    v15 = _mm_loadu_si128(a4);
  }
  if ( v7 )
  {
    sub_157E9D0(v7 + 40, (__int64)v6);
    v9 = *v8;
    v10 = v6[3];
    v6[4] = v8;
    v9 &= 0xFFFFFFFFFFFFFFF8LL;
    v6[3] = v9 | v10 & 7;
    *(_QWORD *)(v9 + 8) = v6 + 3;
    *v8 = *v8 & 7 | (unsigned __int64)(v6 + 3);
  }
  sub_164B780((__int64)v6, v15.m128i_i64);
  v11 = *a1;
  if ( *a1 )
  {
    v17.m128i_i64[0] = *a1;
    sub_1623A60((__int64)&v17, v11, 2);
    v12 = v6[6];
    if ( v12 )
      sub_161E7C0((__int64)(v6 + 6), v12);
    v13 = (unsigned __int8 *)v17.m128i_i64[0];
    v6[6] = v17.m128i_i64[0];
    if ( v13 )
      sub_1623210((__int64)&v17, v13, (__int64)(v6 + 6));
  }
  return v6;
}
