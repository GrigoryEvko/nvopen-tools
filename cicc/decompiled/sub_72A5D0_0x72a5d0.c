// Function: sub_72A5D0
// Address: 0x72a5d0
//
void __fastcall sub_72A5D0(const __m128i *a1, __m128i *a2)
{
  __m128i v2; // xmm5

  *a2 = _mm_loadu_si128(a1);
  a2[1] = _mm_loadu_si128(a1 + 1);
  a2[2] = _mm_loadu_si128(a1 + 2);
  a2[3] = _mm_loadu_si128(a1 + 3);
  a2[4] = _mm_loadu_si128(a1 + 4);
  a2[5] = _mm_loadu_si128(a1 + 5);
  a2[6] = _mm_loadu_si128(a1 + 6);
  a2[7] = _mm_loadu_si128(a1 + 7);
  a2[8] = _mm_loadu_si128(a1 + 8);
  a2[9] = _mm_loadu_si128(a1 + 9);
  a2[10] = _mm_loadu_si128(a1 + 10);
  a2[11] = _mm_loadu_si128(a1 + 11);
  a2[12] = _mm_loadu_si128(a1 + 12);
  v2 = _mm_loadu_si128(a1 + 13);
  a2[5].m128i_i8[10] &= ~0x80u;
  a2[7].m128i_i64[0] = 0;
  a2[6].m128i_i64[0] = 0;
  a2[13] = v2;
  a2[4].m128i_i64[1] = 0;
}
