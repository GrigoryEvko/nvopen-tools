// Function: sub_31FBC20
// Address: 0x31fbc20
//
unsigned __int64 *__fastcall sub_31FBC20(unsigned __int64 *a1, __m128i *a2, unsigned __int64 **a3)
{
  __m128i *v4; // rsi
  __int64 v5; // rcx
  unsigned __int64 *result; // rax

  v4 = (__m128i *)a1[1];
  if ( v4 == (__m128i *)a1[2] )
    return sub_31FB980(a1, v4, a2, a3);
  if ( v4 )
  {
    v4->m128i_i64[0] = (__int64)v4[1].m128i_i64;
    if ( (__m128i *)a2->m128i_i64[0] == &a2[1] )
    {
      v4[1] = _mm_loadu_si128(a2 + 1);
    }
    else
    {
      v4->m128i_i64[0] = a2->m128i_i64[0];
      v4[1].m128i_i64[0] = a2[1].m128i_i64[0];
    }
    v5 = a2->m128i_i64[1];
    a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
    a2->m128i_i64[1] = 0;
    v4->m128i_i64[1] = v5;
    a2[1].m128i_i8[0] = 0;
    result = *a3;
    v4[2].m128i_i64[0] = (__int64)*a3;
    v4 = (__m128i *)a1[1];
  }
  a1[1] = (unsigned __int64)&v4[2].m128i_u64[1];
  return result;
}
