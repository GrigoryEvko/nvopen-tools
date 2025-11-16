// Function: sub_22DD3E0
// Address: 0x22dd3e0
//
unsigned __int64 __fastcall sub_22DD3E0(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v5; // rsi
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int8 *v11; // rdx
  unsigned __int64 v12; // rbx
  __m128i *v13; // r15
  __m128i *v14; // rax
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  __m128i *v17; // rdx
  const __m128i *v18; // rax
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  const __m128i *v22; // [rsp+0h] [rbp-50h]
  size_t v23; // [rsp+10h] [rbp-40h]
  unsigned __int64 v24; // [rsp+18h] [rbp-38h]

  v5 = (const __m128i *)a1[1];
  v6 = *a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v5->m128i_i64 - *a1) >> 3);
  if ( v7 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x3333333333333333LL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 3);
  v11 = &a2->m128i_i8[-v6];
  if ( v9 )
  {
    v20 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v24 = 0;
      v12 = 40;
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x333333333333333LL )
      v10 = 0x333333333333333LL;
    v20 = 40 * v10;
  }
  v22 = a3;
  v21 = sub_22077B0(v20);
  v11 = &a2->m128i_i8[-v6];
  v13 = (__m128i *)v21;
  a3 = v22;
  v24 = v20 + v21;
  v12 = v21 + 40;
LABEL_7:
  v14 = (__m128i *)&v11[(_QWORD)v13];
  if ( &v11[(_QWORD)v13] )
  {
    v15 = _mm_loadu_si128(a3);
    v16 = _mm_loadu_si128(a3 + 1);
    v14[2].m128i_i64[0] = a3[2].m128i_i64[0];
    *v14 = v15;
    v14[1] = v16;
  }
  if ( a2 != (const __m128i *)v6 )
  {
    v17 = v13;
    v18 = (const __m128i *)v6;
    do
    {
      if ( v17 )
      {
        *v17 = _mm_loadu_si128(v18);
        v17[1] = _mm_loadu_si128(v18 + 1);
        v17[2].m128i_i64[0] = v18[2].m128i_i64[0];
      }
      v18 = (const __m128i *)((char *)v18 + 40);
      v17 = (__m128i *)((char *)v17 + 40);
    }
    while ( v18 != a2 );
    v12 = (unsigned __int64)&v13[5] + 8 * (((unsigned __int64)&a2[-3].m128i_u64[1] - v6) >> 3);
  }
  if ( a2 != v5 )
  {
    v23 = 8 * ((unsigned __int64)((char *)v5 - (char *)a2 - 40) >> 3) + 40;
    memcpy((void *)v12, a2, v23);
    v12 += v23;
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  *a1 = (unsigned __int64)v13;
  a1[1] = v12;
  a1[2] = v24;
  return v24;
}
