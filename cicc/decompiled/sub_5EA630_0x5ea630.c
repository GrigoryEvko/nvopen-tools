// Function: sub_5EA630
// Address: 0x5ea630
//
__int64 __fastcall sub_5EA630(__int64 a1, const __m128i *a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 456);
  *(__m128i *)(result + 24) = _mm_loadu_si128(a2);
  *(__m128i *)(result + 40) = _mm_loadu_si128(a2 + 1);
  *(__m128i *)(result + 56) = _mm_loadu_si128(a2 + 2);
  *(__m128i *)(result + 72) = _mm_loadu_si128(a2 + 3);
  *(__m128i *)(result + 88) = _mm_loadu_si128(a2 + 4);
  *(__m128i *)(result + 104) = _mm_loadu_si128(a2 + 5);
  *(_QWORD *)(result + 120) = a2[6].m128i_i64[0];
  return result;
}
