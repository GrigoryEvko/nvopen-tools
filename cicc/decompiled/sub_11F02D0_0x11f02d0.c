// Function: sub_11F02D0
// Address: 0x11f02d0
//
__int64 __fastcall sub_11F02D0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v7; // r14
  unsigned __int64 v8; // rsi
  __int64 v10; // rax
  const __m128i *v11; // rdx
  __int64 v12; // r13
  __int64 result; // rax
  const __m128i *v14; // r12
  const __m128i *v15; // rax
  __int64 m128i_i64; // rcx
  __int64 v17; // rdi
  __m128i *v18; // rdx
  const __m128i *v19; // rsi
  __m128i v20; // xmm0
  __int64 v21; // rsi
  const __m128i *v22; // r15
  const __m128i *v23; // rdi
  int v24; // r15d
  unsigned __int64 v25[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = (const __m128i *)(a1 + 16);
  v8 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x50u, v25, a6);
  v11 = *(const __m128i **)a1;
  v12 = v10;
  result = *(unsigned int *)(a1 + 8);
  v14 = (const __m128i *)(*(_QWORD *)a1 + 80 * result);
  if ( *(const __m128i **)a1 != v14 )
  {
    v15 = v11 + 3;
    m128i_i64 = (__int64)v11[1].m128i_i64;
    v8 = ((char *)v14 - (char *)v11 - 80) & 0xFFFFFFFFFFFFFFF0LL;
    v17 = (__int64)v11[8].m128i_i64 + v8;
    v18 = (__m128i *)v12;
    do
    {
      if ( v18 )
      {
        v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
        v21 = v15[-3].m128i_i64[0];
        if ( v21 == m128i_i64 )
        {
          v18[1] = _mm_loadu_si128(v15 - 2);
        }
        else
        {
          v18->m128i_i64[0] = v21;
          v18[1].m128i_i64[0] = v15[-2].m128i_i64[0];
        }
        v18->m128i_i64[1] = v15[-3].m128i_i64[1];
        v15[-3].m128i_i64[0] = m128i_i64;
        v15[-3].m128i_i64[1] = 0;
        v15[-2].m128i_i8[0] = 0;
        v18[2].m128i_i64[0] = (__int64)v18[3].m128i_i64;
        v19 = (const __m128i *)v15[-1].m128i_i64[0];
        if ( v19 == v15 )
        {
          v18[3] = _mm_loadu_si128(v15);
        }
        else
        {
          v18[2].m128i_i64[0] = (__int64)v19;
          v18[3].m128i_i64[0] = v15->m128i_i64[0];
        }
        v8 = v15[-1].m128i_u64[1];
        v18[2].m128i_i64[1] = v8;
        v20 = _mm_loadu_si128(v15 + 1);
        v15[-1].m128i_i64[0] = (__int64)v15;
        v15[-1].m128i_i64[1] = 0;
        v15->m128i_i8[0] = 0;
        v18[4] = v20;
      }
      v15 += 5;
      v18 += 5;
      m128i_i64 += 80;
    }
    while ( v15 != (const __m128i *)v17 );
    result = *(unsigned int *)(a1 + 8);
    v22 = *(const __m128i **)a1;
    v14 = (const __m128i *)(*(_QWORD *)a1 + 80 * result);
    if ( v14 != *(const __m128i **)a1 )
    {
      do
      {
        v14 -= 5;
        v23 = (const __m128i *)v14[2].m128i_i64[0];
        if ( v23 != &v14[3] )
        {
          v8 = v14[3].m128i_i64[0] + 1;
          j_j___libc_free_0(v23, v8);
        }
        result = (__int64)v14[1].m128i_i64;
        if ( (const __m128i *)v14->m128i_i64[0] != &v14[1] )
        {
          v8 = v14[1].m128i_i64[0] + 1;
          result = j_j___libc_free_0(v14->m128i_i64[0], v8);
        }
      }
      while ( v14 != v22 );
      v14 = *(const __m128i **)a1;
    }
  }
  v24 = v25[0];
  if ( v14 != v7 )
    result = _libc_free(v14, v8);
  *(_QWORD *)a1 = v12;
  *(_DWORD *)(a1 + 12) = v24;
  return result;
}
