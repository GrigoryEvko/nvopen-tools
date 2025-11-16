// Function: sub_1FD3FA0
// Address: 0x1fd3fa0
//
const __m128i *__fastcall sub_1FD3FA0(const __m128i **a1, unsigned __int64 a2)
{
  const __m128i *v3; // rdi
  const __m128i *result; // rax
  const __m128i *v5; // rcx
  __m128i *v6; // r13
  signed __int64 v7; // r12
  __int64 v8; // rax
  __m128i *v9; // rdx

  if ( a2 > 0x333333333333333LL )
    sub_4262D8((__int64)"vector::reserve");
  v3 = *a1;
  result = v3;
  if ( a2 > 0xCCCCCCCCCCCCCCCDLL * (((char *)a1[2] - (char *)v3) >> 3) )
  {
    v5 = a1[1];
    v6 = 0;
    v7 = (char *)v5 - (char *)v3;
    if ( a2 )
    {
      v8 = sub_22077B0(40 * a2);
      v3 = *a1;
      v5 = a1[1];
      v6 = (__m128i *)v8;
      result = *a1;
    }
    if ( v5 != v3 )
    {
      v9 = v6;
      do
      {
        if ( v9 )
        {
          *v9 = _mm_loadu_si128(result);
          v9[1] = _mm_loadu_si128(result + 1);
          v9[2].m128i_i64[0] = result[2].m128i_i64[0];
        }
        result = (const __m128i *)((char *)result + 40);
        v9 = (__m128i *)((char *)v9 + 40);
      }
      while ( v5 != result );
    }
    if ( v3 )
      result = (const __m128i *)j_j___libc_free_0(v3, (char *)a1[2] - (char *)v3);
    *a1 = v6;
    a1[1] = (__m128i *)((char *)v6 + v7);
    a1[2] = (__m128i *)((char *)v6 + 40 * a2);
  }
  return result;
}
