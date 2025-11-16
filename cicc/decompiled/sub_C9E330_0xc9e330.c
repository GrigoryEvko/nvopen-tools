// Function: sub_C9E330
// Address: 0xc9e330
//
__int64 __fastcall sub_C9E330(__int64 a1)
{
  __m128i v2; // xmm0
  __m128i v3; // xmm1

  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  v2 = _mm_loadu_si128((const __m128i *)(a1 + 40));
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  v3 = _mm_loadu_si128((const __m128i *)(a1 + 56));
  *(_WORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(__m128i *)a1 = v2;
  *(__m128i *)(a1 + 16) = v3;
  return 0;
}
