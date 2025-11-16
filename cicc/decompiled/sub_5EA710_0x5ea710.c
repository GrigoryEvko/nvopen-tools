// Function: sub_5EA710
// Address: 0x5ea710
//
__int64 __fastcall sub_5EA710(__int64 a1, __int64 a2, const __m128i *a3, const __m128i *a4)
{
  __int64 v6; // rax
  __m128i v7; // xmm7

  v6 = sub_5E4B20(a1);
  *(_QWORD *)(v6 + 16) = a2;
  *(__m128i *)(v6 + 24) = _mm_loadu_si128(a3);
  *(__m128i *)(v6 + 40) = _mm_loadu_si128(a3 + 1);
  *(__m128i *)(v6 + 56) = _mm_loadu_si128(a3 + 2);
  *(__m128i *)(v6 + 72) = _mm_loadu_si128(a3 + 3);
  *(__m128i *)(v6 + 88) = _mm_loadu_si128(a3 + 4);
  *(__m128i *)(v6 + 104) = _mm_loadu_si128(a3 + 5);
  *(_QWORD *)(v6 + 120) = a3[6].m128i_i64[0];
  *(__m128i *)(v6 + 152) = _mm_loadu_si128(a4);
  v7 = _mm_loadu_si128(a4 + 1);
  *(_BYTE *)(v6 + 184) |= 1u;
  *(__m128i *)(v6 + 168) = v7;
  return sub_5E9580(v6);
}
