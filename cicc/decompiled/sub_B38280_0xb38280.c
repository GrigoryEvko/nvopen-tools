// Function: sub_B38280
// Address: 0xb38280
//
__m128i **__fastcall sub_B38280(__m128i **a1, const __m128i *a2, const char *a3, __int64 a4)
{
  const __m128i *v6; // r13
  const __m128i *v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  signed __int64 v12; // rcx
  __int64 m128i_i64; // rbx
  size_t v14; // rax
  __int8 *v15; // r8
  __m128i *v16; // rcx
  size_t v17; // r14
  __m128i *v18; // rdx
  const void *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  char *v23; // rax
  size_t v24; // rdx
  __m128i *v25; // rcx
  __m128i *v26; // rdi
  __int64 v27; // r8
  __m128i *v28; // r14
  const __m128i *i; // rbx
  const __m128i *v30; // rax
  __int64 v31; // rax
  const __m128i *v32; // rdi
  __int64 v33; // rdi
  const __m128i *v34; // rax
  __m128i *v35; // rdx
  __int64 v36; // rcx
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rax
  __m128i *v41; // rdi
  __m128i *src; // [rsp+10h] [rbp-90h]
  __m128i *v44; // [rsp+18h] [rbp-88h]
  char *v45; // [rsp+18h] [rbp-88h]
  const char *v46; // [rsp+18h] [rbp-88h]
  __int8 *n; // [rsp+20h] [rbp-80h]
  size_t na; // [rsp+20h] [rbp-80h]
  size_t nb; // [rsp+20h] [rbp-80h]
  size_t nc; // [rsp+20h] [rbp-80h]
  __int64 v51; // [rsp+28h] [rbp-78h]
  __m128i *v53; // [rsp+38h] [rbp-68h]
  __int64 v54; // [rsp+48h] [rbp-58h] BYREF
  __m128i *v55; // [rsp+50h] [rbp-50h] BYREF
  size_t v56; // [rsp+58h] [rbp-48h]
  __m128i v57[4]; // [rsp+60h] [rbp-40h] BYREF

  v6 = a1[1];
  v7 = *a1;
  v8 = 0x6DB6DB6DB6DB6DB7LL * (((char *)v6 - (char *)*a1) >> 3);
  if ( v8 == 0x249249249249249LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  if ( v8 )
    v9 = 0x6DB6DB6DB6DB6DB7LL * (((char *)v6 - (char *)v7) >> 3);
  v10 = __CFADD__(v9, v8);
  v11 = v9 + v8;
  v12 = (char *)a2 - (char *)v7;
  if ( v10 )
  {
    v38 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v51 = 0;
      m128i_i64 = 56;
      v53 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x249249249249249LL )
      v11 = 0x249249249249249LL;
    v38 = 56 * v11;
  }
  v46 = a3;
  v39 = sub_22077B0(v38);
  v12 = (char *)a2 - (char *)v7;
  a3 = v46;
  v53 = (__m128i *)v39;
  v51 = v39 + v38;
  m128i_i64 = v39 + 56;
LABEL_7:
  src = (__m128i *)((char *)v53 + v12);
  v55 = v57;
  n = (__int8 *)a3;
  v14 = strlen(a3);
  v15 = n;
  v54 = v14;
  v16 = src;
  v17 = v14;
  if ( v14 > 0xF )
  {
    v40 = sub_22409D0(&v55, &v54, 0);
    v16 = src;
    v15 = n;
    v55 = (__m128i *)v40;
    v41 = (__m128i *)v40;
    v57[0].m128i_i64[0] = v54;
  }
  else
  {
    if ( v14 == 1 )
    {
      v57[0].m128i_i8[0] = *n;
      v18 = v57;
      goto LABEL_10;
    }
    if ( !v14 )
    {
      v56 = 0;
      v57[0].m128i_i8[0] = 0;
      if ( src )
        goto LABEL_11;
LABEL_32:
      v26 = v55;
      goto LABEL_16;
    }
    v41 = v57;
  }
  nc = (size_t)v16;
  memcpy(v41, v15, v17);
  v14 = v54;
  v18 = v55;
  v16 = (__m128i *)nc;
LABEL_10:
  v56 = v14;
  v18->m128i_i8[v14] = 0;
  if ( !v16 )
    goto LABEL_32;
LABEL_11:
  v19 = *(const void **)a4;
  v20 = *(unsigned int *)(a4 + 8);
  v16->m128i_i64[0] = (__int64)v16[1].m128i_i64;
  if ( v55 == v57 )
  {
    v16[1] = _mm_load_si128(v57);
  }
  else
  {
    v16->m128i_i64[0] = (__int64)v55;
    v16[1].m128i_i64[0] = v57[0].m128i_i64[0];
  }
  v21 = v56;
  v22 = 8 * v20;
  v55 = v57;
  v56 = 0;
  v16->m128i_i64[1] = v21;
  v57[0].m128i_i8[0] = 0;
  v16[2].m128i_i64[0] = 0;
  v16[2].m128i_i64[1] = 0;
  v16[3].m128i_i64[0] = 0;
  if ( v22 )
  {
    v44 = v16;
    na = v22;
    v23 = (char *)sub_22077B0(v22);
    v24 = na;
    v25 = v44;
    v44[2].m128i_i64[0] = (__int64)v23;
    v44[3].m128i_i64[0] = (__int64)&v23[na];
    v45 = &v23[na];
    nb = (size_t)v25;
    memcpy(v23, v19, v24);
    v26 = v55;
    v27 = (__int64)v45;
    v16 = (__m128i *)nb;
  }
  else
  {
    v26 = v57;
    v27 = 0;
  }
  v16[2].m128i_i64[1] = v27;
LABEL_16:
  if ( v26 != v57 )
    j_j___libc_free_0(v26, v57[0].m128i_i64[0] + 1);
  if ( a2 != v7 )
  {
    v28 = v53;
    for ( i = v7 + 1; ; i = (const __m128i *)((char *)i + 56) )
    {
      if ( v28 )
      {
        v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
        v30 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v30 == i )
        {
          v28[1] = _mm_loadu_si128(i);
        }
        else
        {
          v28->m128i_i64[0] = (__int64)v30;
          v28[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v28->m128i_i64[1] = i[-1].m128i_i64[1];
        v31 = i[1].m128i_i64[0];
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v28[2].m128i_i64[0] = v31;
        v28[2].m128i_i64[1] = i[1].m128i_i64[1];
        v28[3].m128i_i64[0] = i[2].m128i_i64[0];
        i[2].m128i_i64[0] = 0;
        i[1].m128i_i64[0] = 0;
      }
      else
      {
        v33 = i[1].m128i_i64[0];
        if ( v33 )
          j_j___libc_free_0(v33, i[2].m128i_i64[0] - v33);
      }
      v32 = (const __m128i *)i[-1].m128i_i64[0];
      if ( v32 != i )
        j_j___libc_free_0(v32, i->m128i_i64[0] + 1);
      if ( a2 == (const __m128i *)&i[2].m128i_u64[1] )
        break;
      v28 = (__m128i *)((char *)v28 + 56);
    }
    m128i_i64 = (__int64)v28[7].m128i_i64;
  }
  if ( a2 != v6 )
  {
    v34 = a2;
    v35 = (__m128i *)m128i_i64;
    do
    {
      v35->m128i_i64[0] = (__int64)v35[1].m128i_i64;
      if ( (const __m128i *)v34->m128i_i64[0] == &v34[1] )
      {
        v35[1] = _mm_loadu_si128(v34 + 1);
      }
      else
      {
        v35->m128i_i64[0] = v34->m128i_i64[0];
        v35[1].m128i_i64[0] = v34[1].m128i_i64[0];
      }
      v36 = v34->m128i_i64[1];
      v34 = (const __m128i *)((char *)v34 + 56);
      v35 = (__m128i *)((char *)v35 + 56);
      v35[-3].m128i_i64[0] = v36;
      v35[-2].m128i_i64[1] = v34[-2].m128i_i64[1];
      v35[-1].m128i_i64[0] = v34[-1].m128i_i64[0];
      v35[-1].m128i_i64[1] = v34[-1].m128i_i64[1];
    }
    while ( v34 != v6 );
    m128i_i64 += 56
               * (((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)((char *)v34 - (char *)a2 - 56) >> 3))
                 & 0x1FFFFFFFFFFFFFFFLL)
                + 1);
  }
  if ( v7 )
    j_j___libc_free_0(v7, (char *)a1[2] - (char *)v7);
  *a1 = v53;
  a1[1] = (__m128i *)m128i_i64;
  a1[2] = (__m128i *)v51;
  return a1;
}
