// Function: sub_B56820
// Address: 0xb56820
//
__int64 __fastcall sub_B56820(__int64 a1, __m128i *a2)
{
  const __m128i *v2; // rdx
  __int64 result; // rax
  __int64 v4; // rcx
  const __m128i *v5; // rax
  __m128i *v6; // rcx
  __int64 v7; // rdx
  const __m128i *v8; // rdx
  __int64 v9; // rdx
  const __m128i *v10; // r12
  const __m128i *v11; // rbx
  __int64 v12; // rdi

  v2 = *(const __m128i **)a1;
  result = 7LL * *(unsigned int *)(a1 + 8);
  v4 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v4 )
  {
    v5 = v2 + 1;
    v6 = (__m128i *)((char *)a2
                   + 56
                   * ((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)(v4 - (_QWORD)v2 - 56) >> 3)) & 0x1FFFFFFFFFFFFFFFLL)
                   + 56);
    do
    {
      if ( a2 )
      {
        a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
        v8 = (const __m128i *)v5[-1].m128i_i64[0];
        if ( v8 == v5 )
        {
          a2[1] = _mm_loadu_si128(v5);
        }
        else
        {
          a2->m128i_i64[0] = (__int64)v8;
          a2[1].m128i_i64[0] = v5->m128i_i64[0];
        }
        a2->m128i_i64[1] = v5[-1].m128i_i64[1];
        v7 = v5[1].m128i_i64[0];
        v5[-1].m128i_i64[0] = (__int64)v5;
        v5[-1].m128i_i64[1] = 0;
        v5->m128i_i8[0] = 0;
        a2[2].m128i_i64[0] = v7;
        a2[2].m128i_i64[1] = v5[1].m128i_i64[1];
        a2[3].m128i_i64[0] = v5[2].m128i_i64[0];
        v5[2].m128i_i64[0] = 0;
        v5[1].m128i_i64[1] = 0;
        v5[1].m128i_i64[0] = 0;
      }
      a2 = (__m128i *)((char *)a2 + 56);
      v5 = (const __m128i *)((char *)v5 + 56);
    }
    while ( v6 != a2 );
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(const __m128i **)a1;
    result = 7 * v9;
    v11 = (const __m128i *)(*(_QWORD *)a1 + 56 * v9);
    if ( *(const __m128i **)a1 != v11 )
    {
      do
      {
        v12 = v11[-2].m128i_i64[1];
        v11 = (const __m128i *)((char *)v11 - 56);
        if ( v12 )
          j_j___libc_free_0(v12, v11[3].m128i_i64[0] - v12);
        result = (__int64)v11[1].m128i_i64;
        if ( (const __m128i *)v11->m128i_i64[0] != &v11[1] )
          result = j_j___libc_free_0(v11->m128i_i64[0], v11[1].m128i_i64[0] + 1);
      }
      while ( v11 != v10 );
    }
  }
  return result;
}
