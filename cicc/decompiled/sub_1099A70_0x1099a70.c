// Function: sub_1099A70
// Address: 0x1099a70
//
const __m128i *__fastcall sub_1099A70(__int64 a1, __m128i *a2)
{
  const __m128i *result; // rax
  __int64 v3; // rcx
  __m128i *v4; // rcx
  const __m128i *v5; // rdx
  const __m128i *v6; // r12
  const __m128i *v7; // rbx

  result = *(const __m128i **)a1;
  v3 = 2LL * *(unsigned int *)(a1 + 8);
  if ( v3 * 16 )
  {
    ++result;
    v4 = &a2[v3];
    do
    {
      if ( a2 )
      {
        a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
        v5 = (const __m128i *)result[-1].m128i_i64[0];
        if ( v5 == result )
        {
          a2[1] = _mm_loadu_si128(result);
        }
        else
        {
          a2->m128i_i64[0] = (__int64)v5;
          a2[1].m128i_i64[0] = result->m128i_i64[0];
        }
        a2->m128i_i64[1] = result[-1].m128i_i64[1];
        result[-1].m128i_i64[0] = (__int64)result;
        result[-1].m128i_i64[1] = 0;
        result->m128i_i8[0] = 0;
      }
      a2 += 2;
      result += 2;
    }
    while ( a2 != v4 );
    v6 = *(const __m128i **)a1;
    v7 = (const __m128i *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
    while ( v6 != v7 )
    {
      while ( 1 )
      {
        v7 -= 2;
        result = v7 + 1;
        if ( (const __m128i *)v7->m128i_i64[0] == &v7[1] )
          break;
        result = (const __m128i *)j_j___libc_free_0(v7->m128i_i64[0], v7[1].m128i_i64[0] + 1);
        if ( v6 == v7 )
          return result;
      }
    }
  }
  return result;
}
