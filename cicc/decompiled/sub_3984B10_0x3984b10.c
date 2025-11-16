// Function: sub_3984B10
// Address: 0x3984b10
//
__m128i *__fastcall sub_3984B10(__m128i *a1, __m128i *a2)
{
  __int64 v2; // rcx
  __m128i *result; // rax
  const __m128i *v4; // rdx

  if ( a1 == a2 )
    return a1;
  do
  {
    result = a1;
    a1 += 2;
    if ( a2 == a1 )
      return a2;
    v2 = a1[-2].m128i_i64[0];
  }
  while ( v2 != result[2].m128i_i64[0] );
  if ( a2 != result )
  {
    v4 = result + 4;
    if ( a2 == &result[4] )
      return a1;
    while ( 1 )
    {
      if ( v4->m128i_i64[0] != v2 )
      {
        result += 2;
        *result = _mm_loadu_si128(v4);
        result[1] = _mm_loadu_si128(v4 + 1);
      }
      v4 += 2;
      if ( a2 == v4 )
        break;
      v2 = result->m128i_i64[0];
    }
    result += 2;
  }
  return result;
}
