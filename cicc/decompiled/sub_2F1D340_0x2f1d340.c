// Function: sub_2F1D340
// Address: 0x2f1d340
//
void __fastcall sub_2F1D340(unsigned __int64 *a1, unsigned __int64 a2)
{
  const __m128i *v2; // rbx
  __int64 v3; // r12
  _QWORD *v4; // rdx
  unsigned __int64 v5; // rax
  bool v6; // cf
  unsigned __int64 v7; // rax
  _QWORD *v8; // r12
  const __m128i *v9; // r14
  __int64 m128i_i64; // r13
  __m128i *v11; // r15
  __int64 v12; // r12
  __int64 v13; // rbx
  __m128i v14; // xmm0
  __int64 v15; // rcx
  __m128i v16; // xmm1
  __int64 v17; // rcx
  __m128i v18; // xmm2
  __int64 v19; // rcx
  __m128i v20; // xmm3
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  const __m128i *v25; // rcx
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // [rsp+0h] [rbp-60h]
  unsigned __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  unsigned __int64 v31; // [rsp+20h] [rbp-40h]
  unsigned __int64 v32; // [rsp+28h] [rbp-38h]

  v31 = a2;
  if ( !a2 )
    return;
  v2 = (const __m128i *)*a1;
  v32 = a1[1];
  v3 = v32 - *a1;
  v28 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 6);
  if ( a2 <= 0xAAAAAAAAAAAAAAABLL * ((__int64)(a1[2] - v32) >> 6) )
  {
    v4 = (_QWORD *)a1[1];
    do
    {
      if ( v4 )
      {
        memset(v4, 0, 0xC0u);
        *v4 = v4 + 2;
        v4[6] = v4 + 8;
        v4[12] = v4 + 14;
        v4[18] = v4 + 20;
      }
      v4 += 24;
      --a2;
    }
    while ( a2 );
    a1[1] = 192 * v31 + v32;
    return;
  }
  if ( 0xAAAAAAAAAAAAAALL - v28 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v5 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v32 - *a1) >> 6);
  if ( a2 >= v28 )
    v5 = a2;
  v6 = __CFADD__(v28, v5);
  v7 = v28 + v5;
  if ( v6 )
  {
    v26 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v7 )
    {
      v27 = 0;
      v29 = 0;
      goto LABEL_15;
    }
    if ( v7 > 0xAAAAAAAAAAAAAALL )
      v7 = 0xAAAAAAAAAAAAAALL;
    v26 = 192 * v7;
  }
  v29 = sub_22077B0(v26);
  v2 = (const __m128i *)*a1;
  v27 = v29 + v26;
  v32 = a1[1];
LABEL_15:
  v8 = (_QWORD *)(v29 + v3);
  do
  {
    if ( v8 )
    {
      memset(v8, 0, 0xC0u);
      *v8 = v8 + 2;
      v8[6] = v8 + 8;
      v8[12] = v8 + 14;
      v8[18] = v8 + 20;
    }
    v8 += 24;
    --a2;
  }
  while ( a2 );
  if ( v2 != (const __m128i *)v32 )
  {
    v9 = v2 + 1;
    m128i_i64 = (__int64)v2[4].m128i_i64;
    v11 = (__m128i *)v29;
    v12 = (__int64)v2[7].m128i_i64;
    v13 = (__int64)v2[10].m128i_i64;
    while ( 1 )
    {
      if ( v11 )
      {
        v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
        v25 = (const __m128i *)v9[-1].m128i_i64[0];
        if ( v25 == v9 )
        {
          v11[1] = _mm_loadu_si128(v9);
        }
        else
        {
          v11->m128i_i64[0] = (__int64)v25;
          v11[1].m128i_i64[0] = v9->m128i_i64[0];
        }
        v11->m128i_i64[1] = v9[-1].m128i_i64[1];
        v14 = _mm_loadu_si128(v9 + 1);
        v9[-1].m128i_i64[0] = (__int64)v9;
        v9[-1].m128i_i64[1] = 0;
        v9->m128i_i8[0] = 0;
        v11[3].m128i_i64[0] = (__int64)v11[4].m128i_i64;
        v11[2] = v14;
        v15 = v9[2].m128i_i64[0];
        if ( v15 == m128i_i64 )
        {
          v11[4] = _mm_loadu_si128(v9 + 3);
        }
        else
        {
          v11[3].m128i_i64[0] = v15;
          v11[4].m128i_i64[0] = v9[3].m128i_i64[0];
        }
        v11[3].m128i_i64[1] = v9[2].m128i_i64[1];
        v16 = _mm_loadu_si128(v9 + 4);
        v9[2].m128i_i64[0] = m128i_i64;
        v9[2].m128i_i64[1] = 0;
        v9[3].m128i_i8[0] = 0;
        v11[6].m128i_i64[0] = (__int64)v11[7].m128i_i64;
        v11[5] = v16;
        v17 = v9[5].m128i_i64[0];
        if ( v17 == v12 )
        {
          v11[7] = _mm_loadu_si128(v9 + 6);
        }
        else
        {
          v11[6].m128i_i64[0] = v17;
          v11[7].m128i_i64[0] = v9[6].m128i_i64[0];
        }
        v11[6].m128i_i64[1] = v9[5].m128i_i64[1];
        v18 = _mm_loadu_si128(v9 + 7);
        v9[5].m128i_i64[0] = v12;
        v9[5].m128i_i64[1] = 0;
        v9[6].m128i_i8[0] = 0;
        v11[9].m128i_i64[0] = (__int64)v11[10].m128i_i64;
        v11[8] = v18;
        v19 = v9[8].m128i_i64[0];
        if ( v19 == v13 )
        {
          v11[10] = _mm_loadu_si128(v9 + 9);
        }
        else
        {
          v11[9].m128i_i64[0] = v19;
          v11[10].m128i_i64[0] = v9[9].m128i_i64[0];
        }
        v11[9].m128i_i64[1] = v9[8].m128i_i64[1];
        v20 = _mm_loadu_si128(v9 + 10);
        v9[8].m128i_i64[0] = v13;
        v9[8].m128i_i64[1] = 0;
        v9[9].m128i_i8[0] = 0;
        v11[11] = v20;
      }
      v21 = v9[8].m128i_u64[0];
      if ( v13 != v21 )
        j_j___libc_free_0(v21);
      v22 = v9[5].m128i_u64[0];
      if ( v22 != v12 )
        j_j___libc_free_0(v22);
      v23 = v9[2].m128i_u64[0];
      if ( m128i_i64 != v23 )
        j_j___libc_free_0(v23);
      v24 = v9[-1].m128i_u64[0];
      if ( (const __m128i *)v24 != v9 )
        j_j___libc_free_0(v24);
      v11 += 12;
      m128i_i64 += 192;
      v12 += 192;
      v13 += 192;
      if ( (const __m128i *)v32 == &v9[11] )
        break;
      v9 += 12;
    }
    v32 = *a1;
  }
  if ( v32 )
    j_j___libc_free_0(v32);
  *a1 = v29;
  a1[1] = v29 + 192 * (v28 + v31);
  a1[2] = v27;
}
