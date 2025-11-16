// Function: sub_879080
// Address: 0x879080
//
void __fastcall sub_879080(__m128i *a1, const __m128i *a2, __int64 a3)
{
  if ( a2 )
  {
    *a1 = _mm_loadu_si128(a2);
    a1[1] = _mm_loadu_si128(a2 + 1);
  }
  if ( a3 )
    a1[2].m128i_i64[0] = a3;
}
