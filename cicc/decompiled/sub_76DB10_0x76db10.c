// Function: sub_76DB10
// Address: 0x76db10
//
__m128i *__fastcall sub_76DB10(const __m128i *a1, __int64 a2)
{
  __m128i *v2; // r12
  __m128i v3; // xmm5

  v2 = (__m128i *)sub_726B30(a1[2].m128i_i8[8]);
  *v2 = _mm_loadu_si128(a1);
  v2[1] = _mm_loadu_si128(a1 + 1);
  v2[2] = _mm_loadu_si128(a1 + 2);
  v2[3] = _mm_loadu_si128(a1 + 3);
  v2[4] = _mm_loadu_si128(a1 + 4);
  v3 = _mm_loadu_si128(a1 + 5);
  v2[1].m128i_i64[0] = 0;
  v2[1].m128i_i64[1] = 0;
  v2[5] = v3;
  v2[2].m128i_i8[9] &= ~1u;
  v2[3].m128i_i64[1] = 0;
  v2->m128i_i64[0] = unk_4D03F38;
  v2->m128i_i64[1] = unk_4D03F38;
  sub_7E6810(v2, a2, 0);
  return v2;
}
