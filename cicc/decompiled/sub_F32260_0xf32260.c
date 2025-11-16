// Function: sub_F32260
// Address: 0xf32260
//
__int64 __fastcall sub_F32260(__int64 a1, __int64 m128i_i64, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r12
  __int64 result; // rax
  __int64 v8; // r13
  __m128i *v10; // rbx
  __m128i v11; // xmm0
  __int64 v12; // rdi
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 *v15; // r13
  __int64 *v16; // r12
  __int64 *v17; // rdi
  __int64 *v18; // r15
  __int64 v19; // rbx
  __int64 v20; // rdi
  const __m128i *v21; // [rsp+8h] [rbp-38h]

  v6 = *(const __m128i **)a1;
  result = 9LL * *(unsigned int *)(a1 + 8);
  v8 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v10 = (__m128i *)m128i_i64;
    do
    {
      while ( 1 )
      {
        if ( v10 )
        {
          v11 = _mm_loadu_si128(v6);
          v10[1].m128i_i32[2] = 0;
          v10[1].m128i_i64[0] = (__int64)v10[2].m128i_i64;
          v10[1].m128i_i32[3] = 1;
          *v10 = v11;
          if ( v6[1].m128i_i32[2] )
            break;
        }
        v6 = (const __m128i *)((char *)v6 + 72);
        v10 = (__m128i *)((char *)v10 + 72);
        if ( (const __m128i *)v8 == v6 )
          goto LABEL_7;
      }
      m128i_i64 = (__int64)v6[1].m128i_i64;
      v12 = (__int64)v10[1].m128i_i64;
      v6 = (const __m128i *)((char *)v6 + 72);
      v10 = (__m128i *)((char *)v10 + 72);
      sub_F31AD0(v12, m128i_i64, a3, a4, a5, a6);
    }
    while ( (const __m128i *)v8 != v6 );
LABEL_7:
    result = 9LL * *(unsigned int *)(a1 + 8);
    v21 = *(const __m128i **)a1;
    v13 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
    while ( (const __m128i *)v13 != v21 )
    {
      v14 = *(unsigned int *)(v13 - 48);
      v15 = *(__int64 **)(v13 - 56);
      v13 -= 72;
      v16 = &v15[5 * v14];
      if ( v15 != v16 )
      {
        do
        {
          v16 -= 5;
          v17 = (__int64 *)v16[2];
          if ( v17 != v16 + 5 )
            _libc_free(v17, m128i_i64);
          v18 = (__int64 *)*v16;
          v19 = *v16 + 80LL * *((unsigned int *)v16 + 2);
          if ( *v16 != v19 )
          {
            do
            {
              v19 -= 80;
              v20 = *(_QWORD *)(v19 + 8);
              if ( v20 != v19 + 24 )
                _libc_free(v20, m128i_i64);
            }
            while ( v18 != (__int64 *)v19 );
            v18 = (__int64 *)*v16;
          }
          if ( v18 != v16 + 2 )
            _libc_free(v18, m128i_i64);
        }
        while ( v15 != v16 );
        v15 = *(__int64 **)(v13 + 16);
      }
      result = v13 + 32;
      if ( v15 != (__int64 *)(v13 + 32) )
        result = _libc_free(v15, m128i_i64);
    }
  }
  return result;
}
