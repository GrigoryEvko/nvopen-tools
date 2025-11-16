// Function: sub_38BE440
// Address: 0x38be440
//
__m128i *__fastcall sub_38BE440(__int64 a1, const __m128i *a2)
{
  __m128i *v3; // rax
  _BYTE *v4; // rsi
  __m128i *v5; // r12
  __int64 v6; // rdx
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax

  v3 = (__m128i *)sub_145CBF0((__int64 *)(a1 + 1376), 216, 8);
  v4 = (_BYTE *)a2->m128i_i64[1];
  v5 = v3;
  v6 = (__int64)&v4[a2[1].m128i_i64[0]];
  v3->m128i_i64[0] = (__int64)&unk_49EE580;
  v3->m128i_i64[1] = (__int64)&v3[1].m128i_i64[1];
  sub_38BBC60(&v3->m128i_i64[1], v4, v6);
  v7 = (_BYTE *)a2[4].m128i_i64[0];
  v8 = a2[4].m128i_i64[1];
  v5[2].m128i_i64[1] = a2[2].m128i_i64[1];
  v5[3].m128i_i64[0] = a2[3].m128i_i64[0];
  v5[3].m128i_i64[1] = a2[3].m128i_i64[1];
  v5[4].m128i_i64[0] = (__int64)v5[5].m128i_i64;
  sub_38BBC60(v5[4].m128i_i64, v7, (__int64)&v7[v8]);
  v9 = _mm_loadu_si128(a2 + 6);
  v10 = _mm_loadu_si128(a2 + 7);
  v11 = _mm_loadu_si128(a2 + 12);
  v5[8].m128i_i64[0] = a2[8].m128i_i64[0];
  v12 = a2[8].m128i_i64[1];
  v5[6] = v9;
  v5[8].m128i_i64[1] = v12;
  v13 = a2[9].m128i_i64[0];
  v5[7] = v10;
  v5[9].m128i_i64[0] = v13;
  v14 = a2[9].m128i_i64[1];
  v5[12] = v11;
  v5[9].m128i_i64[1] = v14;
  v5[10].m128i_i64[0] = a2[10].m128i_i64[0];
  v5[10].m128i_i64[1] = a2[10].m128i_i64[1];
  v5[11].m128i_i64[0] = a2[11].m128i_i64[0];
  v5[11].m128i_i64[1] = a2[11].m128i_i64[1];
  v5[13].m128i_i64[0] = a2[13].m128i_i64[0];
  return v5;
}
