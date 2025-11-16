// Function: sub_9D4420
// Address: 0x9d4420
//
__m128i *__fastcall sub_9D4420(const __m128i **a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v4; // rsi
  const __m128i *v5; // r15
  __int64 v6; // rax
  bool v8; // zf
  __int64 v10; // rdi
  __int64 v11; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  char *v14; // rdx
  __int64 m128i_i64; // rbx
  __int64 v16; // r12
  __m128i *v17; // r8
  __m128i *result; // rax
  __m128i v19; // xmm5
  __m128i v20; // xmm6
  __m128i v21; // xmm7
  const __m128i *v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rax
  const __m128i *v25; // [rsp+8h] [rbp-48h]
  __m128i *v26; // [rsp+10h] [rbp-40h]
  __m128i *v27; // [rsp+18h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = ((char *)v4 - (char *)*a1) >> 6;
  if ( v6 == 0x1FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = v6 == 0;
  v10 = ((char *)a1[1] - (char *)*a1) >> 6;
  v11 = 1;
  if ( !v8 )
    v11 = v10;
  v12 = __CFADD__(v10, v11);
  v13 = v10 + v11;
  v14 = (char *)((char *)a2 - (char *)v5);
  if ( v12 )
  {
    v23 = 0x7FFFFFFFFFFFFFC0LL;
  }
  else
  {
    if ( !v13 )
    {
      m128i_i64 = 64;
      v16 = 0;
      v17 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0x1FFFFFFFFFFFFFFLL )
      v13 = 0x1FFFFFFFFFFFFFFLL;
    v23 = v13 << 6;
  }
  v25 = a3;
  v24 = sub_22077B0(v23);
  v14 = (char *)((char *)a2 - (char *)v5);
  a3 = v25;
  v17 = (__m128i *)v24;
  v16 = v24 + v23;
  m128i_i64 = v24 + 64;
LABEL_7:
  result = (__m128i *)&v14[(_QWORD)v17];
  if ( &v14[(_QWORD)v17] )
  {
    v19 = _mm_loadu_si128(a3 + 1);
    v20 = _mm_loadu_si128(a3 + 2);
    v21 = _mm_loadu_si128(a3 + 3);
    *result = _mm_loadu_si128(a3);
    result[1] = v19;
    result[2] = v20;
    result[3] = v21;
  }
  if ( a2 != v5 )
  {
    result = v17;
    v22 = v5;
    do
    {
      if ( result )
      {
        *result = _mm_loadu_si128(v22);
        result[1] = _mm_loadu_si128(v22 + 1);
        result[2] = _mm_loadu_si128(v22 + 2);
        result[3] = _mm_loadu_si128(v22 + 3);
      }
      result += 4;
      v22 += 4;
    }
    while ( result != (__m128i *)&v17->m128i_i8[(char *)a2 - (char *)v5] );
    m128i_i64 = (__int64)result[4].m128i_i64;
  }
  if ( a2 != v4 )
  {
    v26 = v17;
    result = (__m128i *)memcpy((void *)m128i_i64, a2, (char *)v4 - (char *)a2);
    v17 = v26;
    m128i_i64 += (char *)v4 - (char *)a2;
  }
  if ( v5 )
  {
    v27 = v17;
    result = (__m128i *)j_j___libc_free_0(v5, (char *)a1[2] - (char *)v5);
    v17 = v27;
  }
  *a1 = v17;
  a1[1] = (const __m128i *)m128i_i64;
  a1[2] = (const __m128i *)v16;
  return result;
}
