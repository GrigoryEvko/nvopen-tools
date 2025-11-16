// Function: sub_729420
// Address: 0x729420
//
__m128i *__fastcall sub_729420(int a1, const __m128i *a2)
{
  __m128i *result; // rax

  if ( !a2 )
    return 0;
  result = sub_7274B0(a1);
  *result = _mm_loadu_si128(a2 + 1);
  result[1] = _mm_loadu_si128(a2 + 2);
  result[2] = _mm_loadu_si128(a2 + 3);
  result[3].m128i_i64[0] = a2[5].m128i_i64[0];
  return result;
}
