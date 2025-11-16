// Function: sub_727480
// Address: 0x727480
//
void *__fastcall sub_727480(__m128i *a1)
{
  __m128i v2; // xmm0

  a1[3].m128i_i64[0] = 0;
  *a1 = _mm_loadu_si128((const __m128i *)&unk_4F07370);
  v2 = _mm_loadu_si128((const __m128i *)&unk_4F07370);
  a1[1] = v2;
  a1[2] = v2;
  return &unk_4F07370;
}
