// Function: sub_3939AB0
// Address: 0x3939ab0
//
unsigned __int64 *__fastcall sub_3939AB0(
        unsigned __int64 *a1,
        const __m128i *a2,
        unsigned __int64 *a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int8 *v12; // r15
  __int64 m128i_i64; // r13
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // rax
  unsigned __int64 *v20; // r15
  const __m128i *v21; // rbx
  __m128i *i; // r13
  __int64 v23; // rax
  __m128i v24; // xmm1
  unsigned __int64 v25; // r14
  _QWORD *v26; // r12
  _QWORD *v27; // r15
  unsigned __int64 v28; // rdi
  _QWORD *v29; // r12
  _QWORD *v30; // r15
  unsigned __int64 v31; // rdi
  const __m128i *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __m128i v35; // xmm0
  __int64 v36; // rcx
  unsigned __int64 v38; // r13
  const __m128i *v39; // [rsp+8h] [rbp-68h]
  unsigned __int64 v40; // [rsp+10h] [rbp-60h]
  __int64 v42; // [rsp+20h] [rbp-50h]
  unsigned __int64 v43; // [rsp+28h] [rbp-48h]
  _QWORD *v45; // [rsp+38h] [rbp-38h]
  _QWORD *v46; // [rsp+38h] [rbp-38h]

  v39 = (const __m128i *)a1[1];
  v43 = *a1;
  v7 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)((__int64)v39->m128i_i64 - *a1) >> 3);
  if ( v7 == 0x249249249249249LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)((__int64)v39->m128i_i64 - *a1) >> 3);
  v10 = __CFADD__(v8, v7);
  v11 = v8 + v7;
  v12 = &a2->m128i_i8[-v43];
  if ( v10 )
  {
    v38 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v40 = 0;
      m128i_i64 = 56;
      v42 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x249249249249249LL )
      v11 = 0x249249249249249LL;
    v38 = 56 * v11;
  }
  v42 = sub_22077B0(v38);
  v40 = v42 + v38;
  m128i_i64 = v42 + 56;
LABEL_7:
  v14 = *a4;
  v15 = *a3;
  v16 = a3[1];
  v17 = *a5;
  *a5 = 0;
  v18 = a5[1];
  v19 = a5[2];
  a5[1] = 0;
  a5[2] = 0;
  v20 = (unsigned __int64 *)&v12[v42];
  if ( v20 )
  {
    *v20 = v17;
    v20[1] = v18;
    v20[2] = v19;
    v20[3] = 0;
    v20[4] = v15;
    v20[5] = v16;
    v20[6] = v14;
  }
  else if ( v17 )
  {
    j_j___libc_free_0(v17);
  }
  v21 = (const __m128i *)v43;
  if ( a2 != (const __m128i *)v43 )
  {
    for ( i = (__m128i *)v42; ; i = (__m128i *)((char *)i + 56) )
    {
      if ( i )
      {
        i->m128i_i64[0] = v21->m128i_i64[0];
        i->m128i_i64[1] = v21->m128i_i64[1];
        i[1].m128i_i64[0] = v21[1].m128i_i64[0];
        v23 = v21[1].m128i_i64[1];
        v21[1].m128i_i64[0] = 0;
        v21->m128i_i64[1] = 0;
        v21->m128i_i64[0] = 0;
        i[1].m128i_i64[1] = v23;
        v24 = _mm_loadu_si128(v21 + 2);
        v21[1].m128i_i64[1] = 0;
        i[2] = v24;
        i[3].m128i_i64[0] = v21[3].m128i_i64[0];
      }
      v25 = v21[1].m128i_u64[1];
      if ( v25 )
      {
        v26 = *(_QWORD **)(v25 + 24);
        v45 = *(_QWORD **)(v25 + 32);
        if ( v45 != v26 )
        {
          do
          {
            v27 = (_QWORD *)*v26;
            while ( v26 != v27 )
            {
              v28 = (unsigned __int64)v27;
              v27 = (_QWORD *)*v27;
              j_j___libc_free_0(v28);
            }
            v26 += 3;
          }
          while ( v45 != v26 );
          v26 = *(_QWORD **)(v25 + 24);
        }
        if ( v26 )
          j_j___libc_free_0((unsigned __int64)v26);
        v29 = *(_QWORD **)v25;
        v46 = *(_QWORD **)(v25 + 8);
        if ( v46 != *(_QWORD **)v25 )
        {
          do
          {
            v30 = (_QWORD *)*v29;
            while ( v29 != v30 )
            {
              v31 = (unsigned __int64)v30;
              v30 = (_QWORD *)*v30;
              j_j___libc_free_0(v31);
            }
            v29 += 3;
          }
          while ( v46 != v29 );
          v29 = *(_QWORD **)v25;
        }
        if ( v29 )
          j_j___libc_free_0((unsigned __int64)v29);
        j_j___libc_free_0(v25);
      }
      if ( v21->m128i_i64[0] )
        j_j___libc_free_0(v21->m128i_i64[0]);
      v21 = (const __m128i *)((char *)v21 + 56);
      if ( v21 == a2 )
        break;
    }
    m128i_i64 = (__int64)i[7].m128i_i64;
  }
  v32 = a2;
  if ( a2 != v39 )
  {
    v33 = m128i_i64;
    do
    {
      v34 = v32->m128i_i64[0];
      v35 = _mm_loadu_si128(v32 + 2);
      v32 = (const __m128i *)((char *)v32 + 56);
      v33 += 56;
      *(_QWORD *)(v33 - 56) = v34;
      v36 = v32[-3].m128i_i64[0];
      *(__m128i *)(v33 - 24) = v35;
      *(_QWORD *)(v33 - 48) = v36;
      *(_QWORD *)(v33 - 40) = v32[-3].m128i_i64[1];
      *(_QWORD *)(v33 - 32) = v32[-2].m128i_i64[0];
      *(_QWORD *)(v33 - 8) = v32[-1].m128i_i64[1];
    }
    while ( v32 != v39 );
    m128i_i64 += 56
               * (((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)((char *)v32 - (char *)a2 - 56) >> 3))
                 & 0x1FFFFFFFFFFFFFFFLL)
                + 1);
  }
  if ( v43 )
    j_j___libc_free_0(v43);
  *a1 = v42;
  a1[1] = m128i_i64;
  a1[2] = v40;
  return a1;
}
