// Function: sub_2F19A70
// Address: 0x2f19a70
//
unsigned __int64 *__fastcall sub_2F19A70(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  bool v6; // cf
  unsigned __int64 v7; // rax
  __int8 *v8; // rdx
  __int64 m128i_i64; // r12
  __m128i *v10; // rax
  __m128i v11; // xmm2
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  const __m128i *v15; // r15
  __m128i *i; // r12
  __int64 v17; // rax
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r14
  const __m128i *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __m128i v23; // xmm0
  __int64 v24; // rcx
  unsigned __int64 v26; // r12
  __int64 v27; // rax
  unsigned __int64 v28; // [rsp+8h] [rbp-58h]
  unsigned __int64 v29; // [rsp+18h] [rbp-48h]
  __m128i *v30; // [rsp+20h] [rbp-40h]
  const __m128i *v31; // [rsp+28h] [rbp-38h]

  v31 = (const __m128i *)a1[1];
  v29 = *a1;
  v4 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v31->m128i_i64 - *a1) >> 4);
  if ( v4 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v31->m128i_i64 - *a1) >> 4);
  v6 = __CFADD__(v5, v4);
  v7 = v5 - 0x5555555555555555LL * ((__int64)((__int64)v31->m128i_i64 - *a1) >> 4);
  v8 = &a2->m128i_i8[-v29];
  if ( v6 )
  {
    v26 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v7 )
    {
      v28 = 0;
      m128i_i64 = 48;
      v30 = 0;
      goto LABEL_7;
    }
    if ( v7 > 0x2AAAAAAAAAAAAAALL )
      v7 = 0x2AAAAAAAAAAAAAALL;
    v26 = 48 * v7;
  }
  v27 = sub_22077B0(v26);
  v8 = &a2->m128i_i8[-v29];
  v30 = (__m128i *)v27;
  v28 = v27 + v26;
  m128i_i64 = v27 + 48;
LABEL_7:
  v10 = (__m128i *)&v8[(_QWORD)v30];
  if ( &v8[(_QWORD)v30] )
  {
    v11 = _mm_loadu_si128(a3);
    v10[1].m128i_i64[0] = a3[1].m128i_i64[0];
    v12 = a3[1].m128i_i64[1];
    a3[1].m128i_i64[1] = 0;
    v10[1].m128i_i64[1] = v12;
    v13 = a3[2].m128i_i64[0];
    a3[2].m128i_i64[0] = 0;
    v10[2].m128i_i64[0] = v13;
    v14 = a3[2].m128i_i64[1];
    a3[2].m128i_i64[1] = 0;
    v10[2].m128i_i64[1] = v14;
    *v10 = v11;
  }
  v15 = (const __m128i *)v29;
  if ( a2 != (const __m128i *)v29 )
  {
    for ( i = v30; !i; i = (__m128i *)v17 )
    {
      v18 = (unsigned __int64 *)v15[2].m128i_i64[0];
      v19 = (unsigned __int64 *)v15[1].m128i_i64[1];
      if ( v18 != v19 )
      {
        do
        {
          if ( (unsigned __int64 *)*v19 != v19 + 2 )
            j_j___libc_free_0(*v19);
          v19 += 6;
        }
        while ( v18 != v19 );
        v19 = (unsigned __int64 *)v15[1].m128i_i64[1];
      }
      if ( !v19 )
        goto LABEL_12;
      v15 += 3;
      j_j___libc_free_0((unsigned __int64)v19);
      v17 = 48;
      if ( v15 == a2 )
      {
LABEL_22:
        m128i_i64 = (__int64)i[6].m128i_i64;
        goto LABEL_23;
      }
LABEL_13:
      ;
    }
    *i = _mm_loadu_si128(v15);
    i[1].m128i_i64[0] = v15[1].m128i_i64[0];
    i[1].m128i_i64[1] = v15[1].m128i_i64[1];
    i[2].m128i_i64[0] = v15[2].m128i_i64[0];
    i[2].m128i_i64[1] = v15[2].m128i_i64[1];
    v15[2].m128i_i64[1] = 0;
    v15[2].m128i_i64[0] = 0;
    v15[1].m128i_i64[1] = 0;
LABEL_12:
    v15 += 3;
    v17 = (__int64)i[3].m128i_i64;
    if ( v15 == a2 )
      goto LABEL_22;
    goto LABEL_13;
  }
LABEL_23:
  if ( a2 != v31 )
  {
    v20 = a2;
    v21 = m128i_i64;
    do
    {
      v22 = v20[1].m128i_i64[0];
      v23 = _mm_loadu_si128(v20);
      v21 += 48;
      v20 += 3;
      *(_QWORD *)(v21 - 32) = v22;
      v24 = v20[-2].m128i_i64[1];
      *(__m128i *)(v21 - 48) = v23;
      *(_QWORD *)(v21 - 24) = v24;
      *(_QWORD *)(v21 - 16) = v20[-1].m128i_i64[0];
      *(_QWORD *)(v21 - 8) = v20[-1].m128i_i64[1];
    }
    while ( v20 != v31 );
    m128i_i64 += 16
               * (3
                * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v20 - (char *)a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
                + 3);
  }
  if ( v29 )
    j_j___libc_free_0(v29);
  *a1 = (unsigned __int64)v30;
  a1[1] = m128i_i64;
  a1[2] = v28;
  return a1;
}
