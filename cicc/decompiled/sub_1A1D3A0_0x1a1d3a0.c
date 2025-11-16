// Function: sub_1A1D3A0
// Address: 0x1a1d3a0
//
__int64 __fastcall sub_1A1D3A0(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4, const __m128i *a5)
{
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  __int64 v10; // r13
  unsigned __int64 *v11; // r15
  __m128i v12; // xmm0
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  char v19[16]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v20; // [rsp+20h] [rbp-80h]
  __m128i v21; // [rsp+30h] [rbp-70h] BYREF
  __int64 v22; // [rsp+40h] [rbp-60h]
  __m128i v23; // [rsp+50h] [rbp-50h] BYREF
  __int16 v24; // [rsp+60h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u && a4[16] <= 0x10u )
    return sub_15A3950(a2, a3, a4, 0);
  v20 = 257;
  v8 = sub_1648A60(56, 3u);
  v9 = v8;
  if ( v8 )
    sub_15FA660((__int64)v8, (_QWORD *)a2, a3, a4, (__int64)v19, 0);
  v10 = a1[1];
  v11 = (unsigned __int64 *)a1[2];
  if ( a5[1].m128i_i8[0] > 1u )
  {
    v24 = 260;
    v23.m128i_i64[0] = (__int64)(a1 + 8);
    sub_14EC200(&v21, &v23, a5);
  }
  else
  {
    v12 = _mm_loadu_si128(a5);
    v22 = a5[1].m128i_i64[0];
    v21 = v12;
  }
  if ( v10 )
  {
    sub_157E9D0(v10 + 40, (__int64)v9);
    v13 = *v11;
    v14 = v9[3];
    v9[4] = v11;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    v9[3] = v13 | v14 & 7;
    *(_QWORD *)(v13 + 8) = v9 + 3;
    *v11 = *v11 & 7 | (unsigned __int64)(v9 + 3);
  }
  sub_164B780((__int64)v9, v21.m128i_i64);
  v15 = *a1;
  if ( *a1 )
  {
    v23.m128i_i64[0] = *a1;
    sub_1623A60((__int64)&v23, v15, 2);
    v16 = v9[6];
    if ( v16 )
      sub_161E7C0((__int64)(v9 + 6), v16);
    v17 = (unsigned __int8 *)v23.m128i_i64[0];
    v9[6] = v23.m128i_i64[0];
    if ( v17 )
      sub_1623210((__int64)&v23, v17, (__int64)(v9 + 6));
  }
  return (__int64)v9;
}
