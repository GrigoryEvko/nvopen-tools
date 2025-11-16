// Function: sub_F0C210
// Address: 0xf0c210
//
char __fastcall sub_F0C210(const __m128i *a1, int a2, __int64 a3, unsigned __int64 a4, __int64 a5, char a6)
{
  __m128i v6; // xmm1
  unsigned __int64 v7; // xmm2_8
  __m128i v8; // xmm3
  __int64 v9; // rax
  __m128i v11; // xmm1
  unsigned __int64 v12; // xmm2_8
  __m128i v13; // xmm3
  __int64 v14; // rax
  __m128i v15; // xmm5
  unsigned __int64 v16; // xmm6_8
  __m128i v17; // xmm7
  __int64 v18; // rax
  __m128i v19; // xmm5
  unsigned __int64 v20; // xmm6_8
  __m128i v21; // xmm7
  __int64 v22; // rax
  __m128i v23; // [rsp+0h] [rbp-50h] BYREF
  __m128i v24; // [rsp+10h] [rbp-40h]
  unsigned __int64 v25; // [rsp+20h] [rbp-30h]
  __int64 v26; // [rsp+28h] [rbp-28h]
  __m128i v27; // [rsp+30h] [rbp-20h]
  __int64 v28; // [rsp+40h] [rbp-10h]

  switch ( a2 )
  {
    case 15:
      if ( a6 )
      {
        v11 = _mm_loadu_si128(a1 + 7);
        v12 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
        v13 = _mm_loadu_si128(a1 + 9);
        v14 = a1[10].m128i_i64[0];
        v23 = _mm_loadu_si128(a1 + 6);
        v25 = v12;
        v28 = v14;
        v26 = a5;
        v24 = v11;
        v27 = v13;
        return (unsigned int)sub_9AFB10(a3, a4, &v23) == 3;
      }
      else
      {
        v15 = _mm_loadu_si128(a1 + 7);
        v16 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
        v17 = _mm_loadu_si128(a1 + 9);
        v18 = a1[10].m128i_i64[0];
        v23 = _mm_loadu_si128(a1 + 6);
        v25 = v16;
        v28 = v18;
        v26 = a5;
        v24 = v15;
        v27 = v17;
        return (unsigned int)sub_9AC9C0(a3, a4, &v23) == 3;
      }
    case 17:
      if ( a6 )
      {
        v6 = _mm_loadu_si128(a1 + 7);
        v7 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
        v8 = _mm_loadu_si128(a1 + 9);
        v9 = a1[10].m128i_i64[0];
        v23 = _mm_loadu_si128(a1 + 6);
        v25 = v7;
        v28 = v9;
        v26 = a5;
        v24 = v6;
        v27 = v8;
        return (unsigned int)sub_9AF960(a3, a4, &v23) == 3;
      }
      else
      {
        v19 = _mm_loadu_si128(a1 + 7);
        v20 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
        v21 = _mm_loadu_si128(a1 + 9);
        v22 = a1[10].m128i_i64[0];
        v23 = _mm_loadu_si128(a1 + 6);
        v25 = v20;
        v28 = v22;
        v26 = a5;
        v24 = v19;
        v27 = v21;
        return (unsigned int)sub_9AC590(a3, a4, &v23, 0) == 3;
      }
    case 13:
      return sub_F0C000(a1, a3, a4, a5, a6);
    default:
      BUG();
  }
}
