// Function: sub_CC1970
// Address: 0xcc1970
//
__int64 __fastcall sub_CC1970(__int64 a1)
{
  __m128i si128; // xmm0
  __m128i v2; // xmm1

  si128 = _mm_load_si128((const __m128i *)&xmmword_3F6B510);
  v2 = _mm_load_si128((const __m128i *)&xmmword_3F6B500);
  *(_QWORD *)(a1 + 64) = 0;
  *(_WORD *)(a1 + 136) = 0;
  *(__m128i *)(a1 + 16) = si128;
  *(__m128i *)(a1 + 48) = si128;
  *(_BYTE *)(a1 + 138) = 0;
  *(_BYTE *)(a1 + 144) = 0;
  *(__m128i *)a1 = v2;
  *(__m128i *)(a1 + 32) = v2;
  *(_OWORD *)(a1 + 72) = 0;
  *(_OWORD *)(a1 + 88) = 0;
  *(_OWORD *)(a1 + 104) = 0;
  *(_OWORD *)(a1 + 120) = 0;
  return 0;
}
