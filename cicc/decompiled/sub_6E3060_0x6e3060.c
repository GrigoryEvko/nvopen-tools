// Function: sub_6E3060
// Address: 0x6e3060
//
__int64 __fastcall sub_6E3060(const __m128i *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // r8
  __int8 v4; // al

  v1 = sub_6E2F40(0);
  v2 = *(_QWORD *)(v1 + 24);
  v3 = v1;
  *(__m128i *)(v2 + 8) = _mm_loadu_si128(a1);
  *(__m128i *)(v2 + 24) = _mm_loadu_si128(a1 + 1);
  *(__m128i *)(v2 + 40) = _mm_loadu_si128(a1 + 2);
  *(__m128i *)(v2 + 56) = _mm_loadu_si128(a1 + 3);
  *(__m128i *)(v2 + 72) = _mm_loadu_si128(a1 + 4);
  *(__m128i *)(v2 + 88) = _mm_loadu_si128(a1 + 5);
  *(__m128i *)(v2 + 104) = _mm_loadu_si128(a1 + 6);
  *(__m128i *)(v2 + 120) = _mm_loadu_si128(a1 + 7);
  *(__m128i *)(v2 + 136) = _mm_loadu_si128(a1 + 8);
  v4 = a1[1].m128i_i8[0];
  if ( v4 == 2 )
  {
    *(__m128i *)(v2 + 152) = _mm_loadu_si128(a1 + 9);
    *(__m128i *)(v2 + 168) = _mm_loadu_si128(a1 + 10);
    *(__m128i *)(v2 + 184) = _mm_loadu_si128(a1 + 11);
    *(__m128i *)(v2 + 200) = _mm_loadu_si128(a1 + 12);
    *(__m128i *)(v2 + 216) = _mm_loadu_si128(a1 + 13);
    *(__m128i *)(v2 + 232) = _mm_loadu_si128(a1 + 14);
    *(__m128i *)(v2 + 248) = _mm_loadu_si128(a1 + 15);
    *(__m128i *)(v2 + 264) = _mm_loadu_si128(a1 + 16);
    *(__m128i *)(v2 + 280) = _mm_loadu_si128(a1 + 17);
    *(__m128i *)(v2 + 296) = _mm_loadu_si128(a1 + 18);
    *(__m128i *)(v2 + 312) = _mm_loadu_si128(a1 + 19);
    *(__m128i *)(v2 + 328) = _mm_loadu_si128(a1 + 20);
    *(__m128i *)(v2 + 344) = _mm_loadu_si128(a1 + 21);
    return v3;
  }
  else if ( v4 == 5 || v4 == 1 )
  {
    *(_QWORD *)(v2 + 152) = a1[9].m128i_i64[0];
    return v3;
  }
  else
  {
    return v3;
  }
}
