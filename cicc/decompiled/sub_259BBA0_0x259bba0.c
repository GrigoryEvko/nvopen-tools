// Function: sub_259BBA0
// Address: 0x259bba0
//
_BOOL8 __fastcall sub_259BBA0(__int64 a1, __int64 a2)
{
  __m128i *v2; // r14
  unsigned __int8 v4; // al
  unsigned __int64 v5; // rsi
  __m128i v6; // xmm0
  unsigned __int8 *v8; // rax
  char v9; // [rsp+Dh] [rbp-33h] BYREF
  bool v10; // [rsp+Eh] [rbp-32h] BYREF
  char v11; // [rsp+Fh] [rbp-31h] BYREF
  __m128i v12[3]; // [rsp+10h] [rbp-30h] BYREF

  v2 = (__m128i *)(a1 + 72);
  v4 = sub_2509800((_QWORD *)(a1 + 72));
  if ( v4 <= 7u && ((1LL << v4) & 0xA8) != 0 )
  {
    v5 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
    if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
      v5 = *(_QWORD *)(v5 + 24);
    sub_250D230((unsigned __int64 *)v12, v5, 5, 0);
    v6 = _mm_loadu_si128(v12);
  }
  else
  {
    v8 = sub_250CBE0(v2->m128i_i64, a2);
    sub_250D230((unsigned __int64 *)v12, (unsigned __int64)v8, 4, 0);
    v6 = _mm_loadu_si128(v12);
  }
  v12[0] = v6;
  if ( (unsigned __int8)sub_259B8C0(a2, a1, v12, 1, &v9, 0, 0) )
    return sub_2559350(a1, a2);
  if ( (unsigned __int8)sub_252A800(a2, v2, a1, &v10) )
    return sub_2559350(a1, a2);
  v11 = 0;
  if ( (unsigned __int8)sub_2523890(
                          a2,
                          (__int64 (__fastcall *)(__int64, __int64 *))sub_253A610,
                          (__int64)v12,
                          a1,
                          1u,
                          &v11) )
    return sub_2559350(a1, a2);
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  return 0;
}
