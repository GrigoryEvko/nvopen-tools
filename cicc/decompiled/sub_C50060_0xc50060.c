// Function: sub_C50060
// Address: 0xc50060
//
__int64 __fastcall sub_C50060(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  const __m128i *v5; // rdx
  __m128i *v6; // r12
  __int64 result; // rax
  const __m128i *v8; // r14
  const __m128i *v9; // rax
  __m128i *v10; // rdx
  __int64 v11; // rcx
  const __m128i *v12; // rcx
  const __m128i *v13; // r15
  int v14; // r15d
  __int64 v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v4 = sub_C8D7D0(a1, a1 + 16, a2, 40, v15);
  v5 = *(const __m128i **)a1;
  v6 = (__m128i *)v4;
  result = 5LL * *(unsigned int *)(a1 + 8);
  v8 = (const __m128i *)(*(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8));
  if ( *(const __m128i **)a1 != v8 )
  {
    v9 = v5 + 1;
    v3 = (__int64)&v6[2].m128i_i64[((unsigned __int64)((char *)v8 - (char *)v5 - 40) >> 3) + 1];
    v10 = v6;
    do
    {
      if ( v10 )
      {
        v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
        v12 = (const __m128i *)v9[-1].m128i_i64[0];
        if ( v12 == v9 )
        {
          v10[1] = _mm_loadu_si128(v9);
        }
        else
        {
          v10->m128i_i64[0] = (__int64)v12;
          v10[1].m128i_i64[0] = v9->m128i_i64[0];
        }
        v10->m128i_i64[1] = v9[-1].m128i_i64[1];
        v11 = v9[1].m128i_i64[0];
        v9[-1].m128i_i64[0] = (__int64)v9;
        v9[-1].m128i_i64[1] = 0;
        v9->m128i_i8[0] = 0;
        v10[2].m128i_i64[0] = v11;
      }
      v10 = (__m128i *)((char *)v10 + 40);
      v9 = (const __m128i *)((char *)v9 + 40);
    }
    while ( v10 != (__m128i *)v3 );
    v13 = *(const __m128i **)a1;
    result = 5LL * *(unsigned int *)(a1 + 8);
    v8 = (const __m128i *)(*(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8));
    if ( *(const __m128i **)a1 != v8 )
    {
      do
      {
        v8 = (const __m128i *)((char *)v8 - 40);
        result = (__int64)v8[1].m128i_i64;
        if ( (const __m128i *)v8->m128i_i64[0] != &v8[1] )
        {
          v3 = v8[1].m128i_i64[0] + 1;
          result = j_j___libc_free_0(v8->m128i_i64[0], v3);
        }
      }
      while ( v13 != v8 );
      v8 = *(const __m128i **)a1;
    }
  }
  v14 = v15[0];
  if ( (const __m128i *)(a1 + 16) != v8 )
    result = _libc_free(v8, v3);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v14;
  return result;
}
