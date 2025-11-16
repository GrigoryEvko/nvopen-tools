// Function: sub_9D3300
// Address: 0x9d3300
//
__int64 __fastcall sub_9D3300(const __m128i **a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v5; // rbx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rsi
  bool v8; // cf
  unsigned __int64 v9; // rax
  signed __int64 v10; // rsi
  __int64 m128i_i64; // r8
  __m128i *v12; // rax
  __int64 v13; // rsi
  __m128i v14; // xmm2
  __int64 v15; // rsi
  __int64 v16; // rsi
  const __m128i *v17; // r14
  __m128i *i; // r13
  __int64 v19; // rsi
  __int64 v20; // rdi
  const __m128i *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  __m128i v24; // xmm0
  __int64 v25; // rsi
  const __m128i *v26; // rdi
  __int64 v28; // r13
  __int64 v29; // rax
  const __m128i *v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  const __m128i *v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+20h] [rbp-40h]
  __m128i *v34; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v32 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v5 - (char *)*a1) >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v5 - (char *)*a1) >> 3);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x3333333333333333LL * (((char *)v5 - (char *)*a1) >> 3);
  v10 = (char *)a2 - (char *)v32;
  if ( v8 )
  {
    v28 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v31 = 0;
      m128i_i64 = 40;
      v34 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x333333333333333LL )
      v9 = 0x333333333333333LL;
    v28 = 40 * v9;
  }
  v30 = a3;
  v29 = sub_22077B0(v28);
  v10 = (char *)a2 - (char *)v32;
  a3 = v30;
  v34 = (__m128i *)v29;
  m128i_i64 = v29 + 40;
  v31 = v29 + v28;
LABEL_7:
  v12 = (__m128i *)((char *)v34 + v10);
  if ( &v34->m128i_i8[v10] )
  {
    v13 = a3[1].m128i_i64[0];
    v14 = _mm_loadu_si128(a3);
    a3[1].m128i_i64[0] = 0;
    v12[1].m128i_i64[0] = v13;
    v15 = a3[1].m128i_i64[1];
    a3[1].m128i_i64[1] = 0;
    v12[1].m128i_i64[1] = v15;
    v16 = a3[2].m128i_i64[0];
    a3[2].m128i_i64[0] = 0;
    v12[2].m128i_i64[0] = v16;
    *v12 = v14;
  }
  v17 = v32;
  if ( a2 != v32 )
  {
    for ( i = v34; !i; i = (__m128i *)v19 )
    {
      v20 = v17[1].m128i_i64[0];
      if ( !v20 )
        goto LABEL_12;
      j_j___libc_free_0(v20, v17[2].m128i_i64[0] - v20);
      v17 = (const __m128i *)((char *)v17 + 40);
      v19 = 40;
      if ( v17 == a2 )
      {
LABEL_17:
        m128i_i64 = (__int64)i[5].m128i_i64;
        goto LABEL_18;
      }
LABEL_13:
      ;
    }
    *i = _mm_loadu_si128(v17);
    i[1].m128i_i64[0] = v17[1].m128i_i64[0];
    i[1].m128i_i64[1] = v17[1].m128i_i64[1];
    i[2].m128i_i64[0] = v17[2].m128i_i64[0];
    v17[2].m128i_i64[0] = 0;
    v17[1].m128i_i64[0] = 0;
LABEL_12:
    v17 = (const __m128i *)((char *)v17 + 40);
    v19 = (__int64)&i[2].m128i_i64[1];
    if ( v17 == a2 )
      goto LABEL_17;
    goto LABEL_13;
  }
LABEL_18:
  if ( a2 != v5 )
  {
    v21 = a2;
    v22 = m128i_i64;
    do
    {
      v23 = v21[1].m128i_i64[0];
      v24 = _mm_loadu_si128(v21);
      v21 = (const __m128i *)((char *)v21 + 40);
      v22 += 40;
      *(_QWORD *)(v22 - 24) = v23;
      v25 = v21[-1].m128i_i64[0];
      *(__m128i *)(v22 - 40) = v24;
      *(_QWORD *)(v22 - 16) = v25;
      *(_QWORD *)(v22 - 8) = v21[-1].m128i_i64[1];
    }
    while ( v21 != v5 );
    m128i_i64 += 8 * ((unsigned __int64)((char *)v21 - (char *)a2 - 40) >> 3) + 40;
  }
  v26 = v32;
  if ( v32 )
  {
    v33 = m128i_i64;
    j_j___libc_free_0(v26, (char *)a1[2] - (char *)v26);
    m128i_i64 = v33;
  }
  a1[1] = (const __m128i *)m128i_i64;
  *a1 = v34;
  a1[2] = (const __m128i *)v31;
  return v31;
}
