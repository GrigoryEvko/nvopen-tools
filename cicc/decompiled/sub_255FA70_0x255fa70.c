// Function: sub_255FA70
// Address: 0x255fa70
//
__m128i *__fastcall sub_255FA70(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  __int64 v3; // rdx
  __m128i *v5; // rdx
  __m128i v6; // xmm0
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rcx
  __m128i *v10; // r12
  __int64 v11; // rbx

  result = *(__m128i **)a1;
  v3 = 2LL * *(unsigned int *)(a1 + 8);
  if ( v3 * 16 )
  {
    v5 = &a2[v3];
    do
    {
      if ( a2 )
      {
        a2[1].m128i_i64[0] = 0;
        v6 = _mm_loadu_si128(result);
        *result = _mm_loadu_si128(a2);
        *a2 = v6;
        v7 = result[1].m128i_i64[0];
        result[1].m128i_i64[0] = 0;
        v8 = a2[1].m128i_i64[1];
        a2[1].m128i_i64[0] = v7;
        v9 = result[1].m128i_i64[1];
        result[1].m128i_i64[1] = v8;
        a2[1].m128i_i64[1] = v9;
      }
      a2 += 2;
      result += 2;
    }
    while ( v5 != a2 );
    v10 = *(__m128i **)a1;
    v11 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v11 )
    {
      do
      {
        result = *(__m128i **)(v11 - 16);
        v11 -= 32;
        if ( result )
          result = (__m128i *)((__int64 (__fastcall *)(__int64, __int64, __int64))result)(v11, v11, 3);
      }
      while ( (__m128i *)v11 != v10 );
    }
  }
  return result;
}
