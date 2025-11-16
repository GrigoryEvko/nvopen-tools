// Function: sub_1A1C7B0
// Address: 0x1a1c7b0
//
_QWORD *__fastcall sub_1A1C7B0(__int64 *a1, _QWORD *a2, const __m128i *a3)
{
  __int64 v4; // r13
  unsigned __int64 *v5; // r14
  unsigned __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rsi
  unsigned __int8 *v10; // rsi
  __m128i v12; // [rsp+0h] [rbp-70h] BYREF
  __int64 v13; // [rsp+10h] [rbp-60h]
  __m128i v14; // [rsp+20h] [rbp-50h] BYREF
  __int16 v15; // [rsp+30h] [rbp-40h]

  v4 = a1[1];
  v5 = (unsigned __int64 *)a1[2];
  if ( a3[1].m128i_i8[0] > 1u )
  {
    v15 = 260;
    v14.m128i_i64[0] = (__int64)(a1 + 8);
    sub_14EC200(&v12, &v14, a3);
  }
  else
  {
    v13 = a3[1].m128i_i64[0];
    v12 = _mm_loadu_si128(a3);
  }
  if ( v4 )
  {
    sub_157E9D0(v4 + 40, (__int64)a2);
    v6 = *v5;
    v7 = a2[3];
    a2[4] = v5;
    v6 &= 0xFFFFFFFFFFFFFFF8LL;
    a2[3] = v6 | v7 & 7;
    *(_QWORD *)(v6 + 8) = a2 + 3;
    *v5 = *v5 & 7 | (unsigned __int64)(a2 + 3);
  }
  sub_164B780((__int64)a2, v12.m128i_i64);
  v8 = *a1;
  if ( *a1 )
  {
    v14.m128i_i64[0] = *a1;
    sub_1623A60((__int64)&v14, v8, 2);
    v9 = a2[6];
    if ( v9 )
      sub_161E7C0((__int64)(a2 + 6), v9);
    v10 = (unsigned __int8 *)v14.m128i_i64[0];
    a2[6] = v14.m128i_i64[0];
    if ( v10 )
      sub_1623210((__int64)&v14, v10, (__int64)(a2 + 6));
  }
  return a2;
}
