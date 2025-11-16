// Function: sub_2285AF0
// Address: 0x2285af0
//
__int64 __fastcall sub_2285AF0(const __m128i **a1, const __m128i *a2, int a3)
{
  if ( a3 == 1 )
  {
    *a1 = a2;
    return 0;
  }
  else
  {
    if ( a3 == 2 )
      *(__m128i *)a1 = _mm_loadu_si128(a2);
    return 0;
  }
}
