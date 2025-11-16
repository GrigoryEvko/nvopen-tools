// Function: sub_1E949D0
// Address: 0x1e949d0
//
__m128i **__fastcall sub_1E949D0(__m128i **a1, const __m128i *a2, __m128i *a3)
{
  const __m128i *v4; // r15
  const __m128i *v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rsi
  bool v8; // cf
  unsigned __int64 v9; // rax
  signed __int64 v10; // rsi
  __int64 m128i_i64; // rbx
  __m128i *v12; // rax
  __m128i *v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdx
  __m128i *v16; // r12
  const __m128i *i; // rbx
  __int64 v18; // rdx
  const __m128i *v19; // rdi
  const __m128i *v20; // rdx
  __m128i *v21; // rdx
  const __m128i *v22; // rax
  __int64 v23; // rsi
  __int64 v25; // rbx
  __int64 v26; // rax
  __m128i *v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  __m128i *v30; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v4 - (char *)*a1) >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v4 - (char *)v5) >> 3);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x3333333333333333LL * (((char *)v4 - (char *)v5) >> 3);
  v10 = (char *)a2 - (char *)v5;
  if ( v8 )
  {
    v25 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v28 = 0;
      m128i_i64 = 40;
      v30 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x333333333333333LL )
      v9 = 0x333333333333333LL;
    v25 = 40 * v9;
  }
  v27 = a3;
  v26 = sub_22077B0(v25);
  v10 = (char *)a2 - (char *)v5;
  a3 = v27;
  v30 = (__m128i *)v26;
  v28 = v26 + v25;
  m128i_i64 = v26 + 40;
LABEL_7:
  v12 = (__m128i *)((char *)v30 + v10);
  if ( &v30->m128i_i8[v10] )
  {
    v13 = (__m128i *)a3->m128i_i64[0];
    v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
    if ( v13 == &a3[1] )
    {
      v12[1] = _mm_loadu_si128(a3 + 1);
    }
    else
    {
      v12->m128i_i64[0] = (__int64)v13;
      v12[1].m128i_i64[0] = a3[1].m128i_i64[0];
    }
    v14 = a3->m128i_i64[1];
    a3->m128i_i64[0] = (__int64)a3[1].m128i_i64;
    a3->m128i_i64[1] = 0;
    a3[1].m128i_i8[0] = 0;
    v15 = a3[2].m128i_i64[0];
    v12->m128i_i64[1] = v14;
    v12[2].m128i_i64[0] = v15;
  }
  if ( a2 != v5 )
  {
    v16 = v30;
    for ( i = v5 + 1; ; i = (const __m128i *)((char *)i + 40) )
    {
      if ( v16 )
      {
        v16->m128i_i64[0] = (__int64)v16[1].m128i_i64;
        v20 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v20 == i )
        {
          v16[1] = _mm_loadu_si128(i);
        }
        else
        {
          v16->m128i_i64[0] = (__int64)v20;
          v16[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v16->m128i_i64[1] = i[-1].m128i_i64[1];
        v18 = i[1].m128i_i64[0];
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v16[2].m128i_i64[0] = v18;
      }
      v19 = (const __m128i *)i[-1].m128i_i64[0];
      if ( v19 != i )
        j_j___libc_free_0(v19, i->m128i_i64[0] + 1);
      if ( a2 == (const __m128i *)&i[1].m128i_u64[1] )
        break;
      v16 = (__m128i *)((char *)v16 + 40);
    }
    m128i_i64 = (__int64)v16[5].m128i_i64;
  }
  if ( a2 != v4 )
  {
    v21 = (__m128i *)m128i_i64;
    v22 = a2;
    do
    {
      v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
      if ( (const __m128i *)v22->m128i_i64[0] == &v22[1] )
      {
        v21[1] = _mm_loadu_si128(v22 + 1);
      }
      else
      {
        v21->m128i_i64[0] = v22->m128i_i64[0];
        v21[1].m128i_i64[0] = v22[1].m128i_i64[0];
      }
      v23 = v22->m128i_i64[1];
      v22 = (const __m128i *)((char *)v22 + 40);
      v21 = (__m128i *)((char *)v21 + 40);
      v21[-2].m128i_i64[0] = v23;
      v21[-1].m128i_i64[1] = v22[-1].m128i_i64[1];
    }
    while ( v4 != v22 );
    m128i_i64 += 8 * ((unsigned __int64)((char *)v4 - (char *)a2 - 40) >> 3) + 40;
  }
  if ( v5 )
    j_j___libc_free_0(v5, (char *)a1[2] - (char *)v5);
  *a1 = v30;
  a1[1] = (__m128i *)m128i_i64;
  a1[2] = (__m128i *)v28;
  return a1;
}
