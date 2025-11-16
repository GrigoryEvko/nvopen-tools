// Function: sub_70DCF0
// Address: 0x70dcf0
//
const __m128i *__fastcall sub_70DCF0(__int64 a1, __m128i *a2)
{
  const __m128i *result; // rax

  result = *(const __m128i **)(a1 + 176);
  if ( *(_BYTE *)(a1 + 173) == 4 )
  {
    *a2 = _mm_loadu_si128(result);
    a2[1] = _mm_loadu_si128(result + 1);
  }
  else
  {
    *a2 = _mm_loadu_si128(result + 11);
    result = (const __m128i *)result[7].m128i_i64[1];
    a2[1] = _mm_loadu_si128(result + 11);
  }
  return result;
}
