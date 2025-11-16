// Function: sub_95D880
// Address: 0x95d880
//
const __m128i *__fastcall sub_95D880(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  const __m128i *result; // rax
  __int64 v4; // rsi
  const __m128i *v5; // r14
  __m128i *v6; // rdx
  const __m128i *v7; // rcx
  const __m128i *v8; // r15
  int v9; // r15d
  _QWORD v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = sub_C8D7D0(a1, a1 + 16, a2, 32, v10);
  result = *(const __m128i **)a1;
  v4 = 32LL * *(unsigned int *)(a1 + 8);
  v5 = (const __m128i *)(*(_QWORD *)a1 + v4);
  if ( *(const __m128i **)a1 != v5 )
  {
    ++result;
    v4 += v2;
    v6 = (__m128i *)v2;
    do
    {
      if ( v6 )
      {
        v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
        v7 = (const __m128i *)result[-1].m128i_i64[0];
        if ( v7 == result )
        {
          v6[1] = _mm_loadu_si128(result);
        }
        else
        {
          v6->m128i_i64[0] = (__int64)v7;
          v6[1].m128i_i64[0] = result->m128i_i64[0];
        }
        v6->m128i_i64[1] = result[-1].m128i_i64[1];
        result[-1].m128i_i64[0] = (__int64)result;
        result[-1].m128i_i64[1] = 0;
        result->m128i_i8[0] = 0;
      }
      v6 += 2;
      result += 2;
    }
    while ( (__m128i *)v4 != v6 );
    v8 = *(const __m128i **)a1;
    v5 = (const __m128i *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
    if ( *(const __m128i **)a1 != v5 )
    {
      do
      {
        v5 -= 2;
        result = v5 + 1;
        if ( (const __m128i *)v5->m128i_i64[0] != &v5[1] )
        {
          v4 = v5[1].m128i_i64[0] + 1;
          result = (const __m128i *)j_j___libc_free_0(v5->m128i_i64[0], v4);
        }
      }
      while ( v8 != v5 );
      v5 = *(const __m128i **)a1;
    }
  }
  v9 = v10[0];
  if ( (const __m128i *)(a1 + 16) != v5 )
    result = (const __m128i *)_libc_free(v5, v4);
  *(_QWORD *)a1 = v2;
  *(_DWORD *)(a1 + 12) = v9;
  return result;
}
