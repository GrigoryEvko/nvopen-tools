// Function: sub_3828C30
// Address: 0x3828c30
//
__int64 *__fastcall sub_3828C30(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __int64 v7; // rax
  _QWORD *v8; // r13
  __int128 v9; // rax
  unsigned int v11; // [rsp+Ch] [rbp-54h] BYREF
  __m128i v12; // [rsp+10h] [rbp-50h] BYREF
  __m128i v13; // [rsp+20h] [rbp-40h] BYREF
  __int64 v14; // [rsp+30h] [rbp-30h] BYREF
  int v15; // [rsp+38h] [rbp-28h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = _mm_loadu_si128((const __m128i *)v3);
  v6 = _mm_loadu_si128((const __m128i *)(v3 + 40));
  v14 = v4;
  v7 = *(_QWORD *)(v3 + 80);
  v12 = v5;
  LODWORD(v7) = *(_DWORD *)(v7 + 96);
  v13 = v6;
  v11 = v7;
  if ( v4 )
    sub_B96E90((__int64)&v14, v4, 1);
  v15 = *(_DWORD *)(a2 + 72);
  sub_3827AB0(a1, (unsigned __int64 *)&v12, (__int64)&v13, &v11, (__int64)&v14, v5);
  if ( v14 )
    sub_B91220((__int64)&v14, v14);
  if ( !v13.m128i_i64[0] )
    return (__int64 *)v12.m128i_i64[0];
  v8 = (_QWORD *)a1[1];
  *(_QWORD *)&v9 = sub_33ED040(v8, v11);
  return sub_33EC3B0(v8, (__int64 *)a2, v12.m128i_i64[0], v12.m128i_i64[1], v13.m128i_i64[0], v13.m128i_i64[1], v9);
}
