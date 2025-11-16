// Function: sub_3225850
// Address: 0x3225850
//
__m128i *__fastcall sub_3225850(__m128i **a1, __m128i *a2)
{
  __m128i *result; // rax
  __m128i *v4; // rsi
  __int64 v5; // rsi

  result = a1[1];
  if ( result == a1[2] )
  {
    sub_8F99A0(a1, a1[1], a2);
    return a1[1] - 2;
  }
  else
  {
    if ( result )
    {
      result->m128i_i64[0] = (__int64)result[1].m128i_i64;
      v4 = (__m128i *)a2->m128i_i64[0];
      if ( v4 == &a2[1] )
      {
        result[1] = _mm_loadu_si128(a2 + 1);
      }
      else
      {
        result->m128i_i64[0] = (__int64)v4;
        result[1].m128i_i64[0] = a2[1].m128i_i64[0];
      }
      v5 = a2->m128i_i64[1];
      a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
      a2->m128i_i64[1] = 0;
      result->m128i_i64[1] = v5;
      a2[1].m128i_i8[0] = 0;
      result = a1[1];
    }
    a1[1] = result + 2;
  }
  return result;
}
