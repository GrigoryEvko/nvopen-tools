// Function: sub_643D80
// Address: 0x643d80
//
void __fastcall sub_643D80(__int64 a1)
{
  const __m128i *v1; // rax
  __m128i *v2; // rdx

  if ( *(_QWORD *)(a1 + 304) )
  {
    v1 = (const __m128i *)sub_72C930(a1);
    v2 = *(__m128i **)(a1 + 304);
    *v2 = _mm_loadu_si128(v1);
    v2[1] = _mm_loadu_si128(v1 + 1);
    v2[2] = _mm_loadu_si128(v1 + 2);
    v2[3] = _mm_loadu_si128(v1 + 3);
    v2[4] = _mm_loadu_si128(v1 + 4);
    v2[5] = _mm_loadu_si128(v1 + 5);
    v2[6] = _mm_loadu_si128(v1 + 6);
    v2[7] = _mm_loadu_si128(v1 + 7);
    v2[8] = _mm_loadu_si128(v1 + 8);
    v2[9] = _mm_loadu_si128(v1 + 9);
    v2[10] = _mm_loadu_si128(v1 + 10);
    v2[11] = _mm_loadu_si128(v1 + 11);
  }
  *(_WORD *)(a1 + 124) &= 0xF07Fu;
  *(_QWORD *)(a1 + 304) = 0;
}
