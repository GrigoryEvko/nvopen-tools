// Function: sub_82BB50
// Address: 0x82bb50
//
_QWORD *__fastcall sub_82BB50(const __m128i *a1)
{
  _QWORD *v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int8 v7; // al
  __int64 v8; // rdi
  _QWORD *v9; // [rsp+8h] [rbp-178h] BYREF
  _OWORD v10[9]; // [rsp+10h] [rbp-170h] BYREF
  __m128i v11; // [rsp+A0h] [rbp-E0h]
  __m128i v12; // [rsp+B0h] [rbp-D0h]
  __m128i v13; // [rsp+C0h] [rbp-C0h]
  __m128i v14; // [rsp+D0h] [rbp-B0h]
  __m128i v15; // [rsp+E0h] [rbp-A0h]
  __m128i v16; // [rsp+F0h] [rbp-90h]
  __m128i v17; // [rsp+100h] [rbp-80h]
  __m128i v18; // [rsp+110h] [rbp-70h]
  __m128i v19; // [rsp+120h] [rbp-60h]
  __m128i v20; // [rsp+130h] [rbp-50h]
  __m128i v21; // [rsp+140h] [rbp-40h]
  __m128i v22; // [rsp+150h] [rbp-30h]
  __m128i v23; // [rsp+160h] [rbp-20h]

  v2 = sub_724DC0();
  v9 = v2;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u
    && a1[1].m128i_i8[0] == 1
    && sub_6ED310((__int64 *)a1[9].m128i_i64[0], (__int64)v2, qword_4D03C50, v3, v4, v5) )
  {
    v10[0] = _mm_loadu_si128(a1);
    v7 = a1[1].m128i_i8[0];
    v10[1] = _mm_loadu_si128(a1 + 1);
    v10[2] = _mm_loadu_si128(a1 + 2);
    v10[3] = _mm_loadu_si128(a1 + 3);
    v10[4] = _mm_loadu_si128(a1 + 4);
    v10[5] = _mm_loadu_si128(a1 + 5);
    v10[6] = _mm_loadu_si128(a1 + 6);
    v10[7] = _mm_loadu_si128(a1 + 7);
    v10[8] = _mm_loadu_si128(a1 + 8);
    if ( v7 == 2 )
    {
      v11 = _mm_loadu_si128(a1 + 9);
      v12 = _mm_loadu_si128(a1 + 10);
      v13 = _mm_loadu_si128(a1 + 11);
      v14 = _mm_loadu_si128(a1 + 12);
      v15 = _mm_loadu_si128(a1 + 13);
      v16 = _mm_loadu_si128(a1 + 14);
      v17 = _mm_loadu_si128(a1 + 15);
      v18 = _mm_loadu_si128(a1 + 16);
      v19 = _mm_loadu_si128(a1 + 17);
      v20 = _mm_loadu_si128(a1 + 18);
      v21 = _mm_loadu_si128(a1 + 19);
      v22 = _mm_loadu_si128(a1 + 20);
      v23 = _mm_loadu_si128(a1 + 21);
    }
    else if ( v7 == 5 || v7 == 1 )
    {
      v11.m128i_i64[0] = a1[9].m128i_i64[0];
    }
    v8 = (__int64)v9;
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
      v9[18] = a1[9].m128i_i64[0];
    sub_6E6A50(v8, (__int64)a1);
    sub_6E4BC0((__int64)a1, (__int64)v10);
  }
  return sub_724E30((__int64)&v9);
}
