// Function: sub_257C0C0
// Address: 0x257c0c0
//
__int64 __fastcall sub_257C0C0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  unsigned __int8 v4; // al
  unsigned __int64 v5; // rsi
  __m128i v6; // xmm0
  unsigned __int8 *v8; // rax
  unsigned __int64 v9; // rax
  char v10; // [rsp+Fh] [rbp-31h] BYREF
  __m128i v11; // [rsp+10h] [rbp-30h] BYREF

  v2 = (__int64 *)(a1 + 72);
  v4 = sub_2509800((_QWORD *)(a1 + 72));
  if ( v4 <= 7u && ((1LL << v4) & 0xA8) != 0 )
  {
    v5 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
    if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
      v5 = *(_QWORD *)(v5 + 24);
    sub_250D230((unsigned __int64 *)&v11, v5, 5, 0);
    v6 = _mm_loadu_si128(&v11);
  }
  else
  {
    v8 = sub_250CBE0(v2, a2);
    sub_250D230((unsigned __int64 *)&v11, (unsigned __int64)v8, 4, 0);
    v6 = _mm_loadu_si128(&v11);
  }
  v11 = v6;
  if ( (unsigned __int8)sub_257BF90(a2, a1, &v11, 1, &v10, 0, 0) )
    return 1;
  v9 = sub_250D070(v2);
  v11.m128i_i64[0] = a2;
  v11.m128i_i64[1] = a1;
  if ( (unsigned __int8)sub_252FFB0(
                          a2,
                          (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_257C1F0,
                          (__int64)&v11,
                          a1,
                          v9,
                          0,
                          1,
                          1,
                          0,
                          0) )
    return 1;
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  return 0;
}
