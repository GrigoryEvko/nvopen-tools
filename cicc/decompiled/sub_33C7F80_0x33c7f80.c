// Function: sub_33C7F80
// Address: 0x33c7f80
//
__int64 __fastcall sub_33C7F80(const __m128i **a1, const __m128i *a2, int a3)
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
