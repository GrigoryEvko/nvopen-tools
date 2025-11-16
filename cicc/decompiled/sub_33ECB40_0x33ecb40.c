// Function: sub_33ECB40
// Address: 0x33ecb40
//
unsigned __int64 __fastcall sub_33ECB40(unsigned __int64 *a1, const __m128i *a2)
{
  __m128i *v3; // rsi
  unsigned __int64 result; // rax

  v3 = (__m128i *)a1[1];
  if ( v3 == (__m128i *)a1[2] )
    return sub_337CFB0(a1, v3, a2);
  if ( v3 )
  {
    *v3 = _mm_loadu_si128(a2);
    v3[1] = _mm_loadu_si128(a2 + 1);
    v3[2] = _mm_loadu_si128(a2 + 2);
    v3 = (__m128i *)a1[1];
  }
  a1[1] = (unsigned __int64)&v3[3];
  return result;
}
