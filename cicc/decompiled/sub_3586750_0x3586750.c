// Function: sub_3586750
// Address: 0x3586750
//
void __fastcall sub_3586750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // r8
  const __m128i *v12; // r9
  __m128i *v13; // rbx
  const __m128i *v14; // r14
  const __m128i *v15; // rax
  __int64 m128i_i64; // rdx
  const __m128i *v17; // rcx
  __m128i v18; // xmm0
  __int64 v19; // rcx
  unsigned __int64 *v20; // r13
  unsigned __int64 *v21; // rbx
  unsigned __int64 v22; // rdi
  unsigned __int64 *v23; // r14
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rax
  unsigned __int64 *v26; // rbx
  unsigned __int64 v27; // rdi
  unsigned __int64 *v28; // r13
  unsigned __int64 *v29; // rbx
  unsigned __int64 v30; // rdi
  __m128i *v31; // r15
  __int64 v32; // r9
  __m128i *v33; // r14
  __m128i *v34; // rbx
  __int64 v35; // rdx
  __m128i *v36; // rsi
  __m128i *v37; // rdi
  __int64 v38; // rdx
  __m128i *v39; // rax
  __m128i *v40; // rdi
  size_t v41; // rdx
  size_t v42; // rdx
  unsigned __int64 *v43; // r14
  unsigned __int64 v44; // rdi
  __m128i *v45; // rcx
  __m128i *v46; // r14
  __int64 v47; // r15
  __int64 v48; // rdx
  __m128i *v49; // rsi
  __m128i *v50; // rdi
  __int64 v51; // rdx
  __m128i *v52; // rax
  __m128i *v53; // rdi
  size_t v54; // rdx
  size_t v55; // rdx
  __int64 v56; // [rsp-50h] [rbp-50h]
  __int64 v57; // [rsp-50h] [rbp-50h]
  __m128i *v58; // [rsp-50h] [rbp-50h]
  __m128i *v59; // [rsp-50h] [rbp-50h]
  int v60; // [rsp-44h] [rbp-44h]
  unsigned __int64 *v61; // [rsp-40h] [rbp-40h]
  unsigned __int64 v62; // [rsp-40h] [rbp-40h]
  unsigned __int64 v63; // [rsp-40h] [rbp-40h]
  unsigned __int64 v64; // [rsp-40h] [rbp-40h]
  unsigned __int64 v65; // [rsp-40h] [rbp-40h]
  __int64 v66; // [rsp-40h] [rbp-40h]

  if ( a1 == a2 )
    return;
  v8 = *(unsigned __int64 **)a1;
  v9 = *(unsigned int *)(a1 + 8);
  v61 = (unsigned __int64 *)(a2 + 16);
  v10 = *(_QWORD *)a1;
  if ( *(_QWORD *)a2 != a2 + 16 )
  {
    v23 = &v8[10 * v9];
    if ( v23 != v8 )
    {
      do
      {
        v23 -= 10;
        v24 = v23[4];
        if ( (unsigned __int64 *)v24 != v23 + 6 )
          j_j___libc_free_0(v24);
        if ( (unsigned __int64 *)*v23 != v23 + 2 )
          j_j___libc_free_0(*v23);
      }
      while ( v23 != v8 );
      v10 = *(_QWORD *)a1;
    }
    if ( v10 != a1 + 16 )
      _libc_free(v10);
    *(_QWORD *)a1 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
    *(_QWORD *)(a2 + 8) = 0;
    *(_QWORD *)a2 = v61;
    return;
  }
  v11 = *(unsigned int *)(a2 + 8);
  v60 = *(_DWORD *)(a2 + 8);
  if ( v11 <= v9 )
  {
    v25 = *(_QWORD *)a1;
    if ( !*(_DWORD *)(a2 + 8) )
    {
LABEL_35:
      v26 = (unsigned __int64 *)(v25 + 80 * v9);
      while ( (unsigned __int64 *)v10 != v26 )
      {
        v26 -= 10;
        v27 = v26[4];
        if ( (unsigned __int64 *)v27 != v26 + 6 )
          j_j___libc_free_0(v27);
        if ( (unsigned __int64 *)*v26 != v26 + 2 )
          j_j___libc_free_0(*v26);
      }
      *(_DWORD *)(a1 + 8) = v60;
      v28 = *(unsigned __int64 **)a2;
      v29 = (unsigned __int64 *)(*(_QWORD *)a2 + 80LL * *(unsigned int *)(a2 + 8));
      if ( *(unsigned __int64 **)a2 != v29 )
      {
        do
        {
          v29 -= 10;
          v30 = v29[4];
          if ( (unsigned __int64 *)v30 != v29 + 6 )
            j_j___libc_free_0(v30);
          if ( (unsigned __int64 *)*v29 != v29 + 2 )
            j_j___libc_free_0(*v29);
        }
        while ( v28 != v29 );
      }
      goto LABEL_23;
    }
    v45 = (__m128i *)(v8 + 6);
    v46 = (__m128i *)(a2 + 32);
    v66 = 10 * v11;
    v47 = (__int64)&v8[10 * v11 + 6];
    while ( 1 )
    {
      v52 = (__m128i *)v46[-1].m128i_i64[0];
      v53 = (__m128i *)v45[-3].m128i_i64[0];
      if ( v52 == v46 )
      {
        v54 = v46[-1].m128i_u64[1];
        if ( v54 )
        {
          if ( v54 == 1 )
          {
            v53->m128i_i8[0] = v46->m128i_i8[0];
            v54 = v46[-1].m128i_u64[1];
            v53 = (__m128i *)v45[-3].m128i_i64[0];
          }
          else
          {
            v58 = v45;
            memcpy(v53, v46, v54);
            v45 = v58;
            v54 = v46[-1].m128i_u64[1];
            v53 = (__m128i *)v58[-3].m128i_i64[0];
          }
        }
        v45[-3].m128i_i64[1] = v54;
        v53->m128i_i8[v54] = 0;
        v53 = (__m128i *)v46[-1].m128i_i64[0];
      }
      else
      {
        if ( v53 == &v45[-2] )
        {
          v45[-3].m128i_i64[0] = (__int64)v52;
          v45[-3].m128i_i64[1] = v46[-1].m128i_i64[1];
          v45[-2].m128i_i64[0] = v46->m128i_i64[0];
        }
        else
        {
          v45[-3].m128i_i64[0] = (__int64)v52;
          v48 = v45[-2].m128i_i64[0];
          v45[-3].m128i_i64[1] = v46[-1].m128i_i64[1];
          v45[-2].m128i_i64[0] = v46->m128i_i64[0];
          if ( v53 )
          {
            v46[-1].m128i_i64[0] = (__int64)v53;
            v46->m128i_i64[0] = v48;
            goto LABEL_83;
          }
        }
        v46[-1].m128i_i64[0] = (__int64)v46;
        v53 = v46;
      }
LABEL_83:
      v46[-1].m128i_i64[1] = 0;
      v53->m128i_i8[0] = 0;
      v49 = (__m128i *)v46[1].m128i_i64[0];
      v50 = (__m128i *)v45[-1].m128i_i64[0];
      if ( v49 == &v46[2] )
      {
        v55 = v46[1].m128i_u64[1];
        if ( v55 )
        {
          if ( v55 == 1 )
          {
            v50->m128i_i8[0] = v46[2].m128i_i8[0];
            v55 = v46[1].m128i_u64[1];
            v50 = (__m128i *)v45[-1].m128i_i64[0];
          }
          else
          {
            v59 = v45;
            memcpy(v50, v49, v55);
            v45 = v59;
            v55 = v46[1].m128i_u64[1];
            v50 = (__m128i *)v59[-1].m128i_i64[0];
          }
        }
        v45[-1].m128i_i64[1] = v55;
        v50->m128i_i8[v55] = 0;
        v50 = (__m128i *)v46[1].m128i_i64[0];
        goto LABEL_87;
      }
      if ( v45 == v50 )
      {
        v45[-1].m128i_i64[0] = (__int64)v49;
        v45[-1].m128i_i64[1] = v46[1].m128i_i64[1];
        v45->m128i_i64[0] = v46[2].m128i_i64[0];
LABEL_98:
        v46[1].m128i_i64[0] = (__int64)v46[2].m128i_i64;
        v50 = v46 + 2;
        goto LABEL_87;
      }
      v45[-1].m128i_i64[0] = (__int64)v49;
      v51 = v45->m128i_i64[0];
      v45[-1].m128i_i64[1] = v46[1].m128i_i64[1];
      v45->m128i_i64[0] = v46[2].m128i_i64[0];
      if ( !v50 )
        goto LABEL_98;
      v46[1].m128i_i64[0] = (__int64)v50;
      v46[2].m128i_i64[0] = v51;
LABEL_87:
      v46[1].m128i_i64[1] = 0;
      v45 += 5;
      v46 += 5;
      v50->m128i_i8[0] = 0;
      v45[-4] = _mm_loadu_si128(v46 - 2);
      if ( v45 == (__m128i *)v47 )
      {
        v25 = *(_QWORD *)a1;
        v9 = *(unsigned int *)(a1 + 8);
        v10 = (unsigned __int64)&v8[v66];
        goto LABEL_35;
      }
    }
  }
  if ( v11 > *(unsigned int *)(a1 + 12) )
  {
    v43 = &v8[10 * v9];
    while ( v43 != v8 )
    {
      while ( 1 )
      {
        v43 -= 10;
        v44 = v43[4];
        if ( (unsigned __int64 *)v44 != v43 + 6 )
        {
          v64 = v11;
          j_j___libc_free_0(v44);
          v11 = v64;
        }
        if ( (unsigned __int64 *)*v43 == v43 + 2 )
          break;
        v65 = v11;
        j_j___libc_free_0(*v43);
        v11 = v65;
        if ( v43 == v8 )
          goto LABEL_77;
      }
    }
LABEL_77:
    *(_DWORD *)(a1 + 8) = 0;
    sub_11F02D0(a1, v11, a3, v9, v11, a6);
    v11 = *(unsigned int *)(a2 + 8);
    v9 = 0;
    v8 = *(unsigned __int64 **)a1;
    v61 = *(unsigned __int64 **)a2;
    v12 = *(const __m128i **)a2;
    goto LABEL_6;
  }
  v12 = (const __m128i *)(a2 + 16);
  if ( !*(_DWORD *)(a1 + 8) )
    goto LABEL_6;
  v31 = (__m128i *)(v8 + 6);
  v32 = 80 * v9;
  v33 = (__m128i *)(a2 + 32);
  v9 = v32;
  v34 = (__m128i *)((char *)v8 + v32 + 48);
  do
  {
    v39 = (__m128i *)v33[-1].m128i_i64[0];
    v40 = (__m128i *)v31[-3].m128i_i64[0];
    if ( v39 == v33 )
    {
      v41 = v33[-1].m128i_u64[1];
      if ( v41 )
      {
        if ( v41 == 1 )
        {
          v40->m128i_i8[0] = v33->m128i_i8[0];
          v41 = v33[-1].m128i_u64[1];
          v40 = (__m128i *)v31[-3].m128i_i64[0];
        }
        else
        {
          v56 = v32;
          v62 = v9;
          memcpy(v40, v33, v41);
          v41 = v33[-1].m128i_u64[1];
          v40 = (__m128i *)v31[-3].m128i_i64[0];
          v32 = v56;
          v9 = v62;
        }
      }
      v31[-3].m128i_i64[1] = v41;
      v40->m128i_i8[v41] = 0;
      v40 = (__m128i *)v33[-1].m128i_i64[0];
    }
    else
    {
      if ( v40 == &v31[-2] )
      {
        v31[-3].m128i_i64[0] = (__int64)v39;
        v31[-3].m128i_i64[1] = v33[-1].m128i_i64[1];
        v31[-2].m128i_i64[0] = v33->m128i_i64[0];
      }
      else
      {
        v31[-3].m128i_i64[0] = (__int64)v39;
        v35 = v31[-2].m128i_i64[0];
        v31[-3].m128i_i64[1] = v33[-1].m128i_i64[1];
        v31[-2].m128i_i64[0] = v33->m128i_i64[0];
        if ( v40 )
        {
          v33[-1].m128i_i64[0] = (__int64)v40;
          v33->m128i_i64[0] = v35;
          goto LABEL_52;
        }
      }
      v33[-1].m128i_i64[0] = (__int64)v33;
      v40 = v33;
    }
LABEL_52:
    v33[-1].m128i_i64[1] = 0;
    v40->m128i_i8[0] = 0;
    v36 = (__m128i *)v33[1].m128i_i64[0];
    v37 = (__m128i *)v31[-1].m128i_i64[0];
    if ( v36 == &v33[2] )
    {
      v42 = v33[1].m128i_u64[1];
      if ( v42 )
      {
        if ( v42 == 1 )
        {
          v37->m128i_i8[0] = v33[2].m128i_i8[0];
          v42 = v33[1].m128i_u64[1];
          v37 = (__m128i *)v31[-1].m128i_i64[0];
        }
        else
        {
          v57 = v32;
          v63 = v9;
          memcpy(v37, v36, v42);
          v42 = v33[1].m128i_u64[1];
          v37 = (__m128i *)v31[-1].m128i_i64[0];
          v32 = v57;
          v9 = v63;
        }
      }
      v31[-1].m128i_i64[1] = v42;
      v37->m128i_i8[v42] = 0;
      v37 = (__m128i *)v33[1].m128i_i64[0];
    }
    else
    {
      if ( v37 == v31 )
      {
        v31[-1].m128i_i64[0] = (__int64)v36;
        v31[-1].m128i_i64[1] = v33[1].m128i_i64[1];
        v31->m128i_i64[0] = v33[2].m128i_i64[0];
      }
      else
      {
        v31[-1].m128i_i64[0] = (__int64)v36;
        v38 = v31->m128i_i64[0];
        v31[-1].m128i_i64[1] = v33[1].m128i_i64[1];
        v31->m128i_i64[0] = v33[2].m128i_i64[0];
        if ( v37 )
        {
          v33[1].m128i_i64[0] = (__int64)v37;
          v33[2].m128i_i64[0] = v38;
          goto LABEL_56;
        }
      }
      v33[1].m128i_i64[0] = (__int64)v33[2].m128i_i64;
      v37 = v33 + 2;
    }
LABEL_56:
    v33[1].m128i_i64[1] = 0;
    v31 += 5;
    v33 += 5;
    v37->m128i_i8[0] = 0;
    v31[-4] = _mm_loadu_si128(v33 - 2);
  }
  while ( v31 != v34 );
  v11 = *(unsigned int *)(a2 + 8);
  v8 = *(unsigned __int64 **)a1;
  v61 = *(unsigned __int64 **)a2;
  v12 = (const __m128i *)(*(_QWORD *)a2 + v32);
LABEL_6:
  v13 = (__m128i *)((char *)v8 + v9);
  v14 = (const __m128i *)&v61[10 * v11];
  if ( v14 != v12 )
  {
    v15 = v12 + 3;
    m128i_i64 = (__int64)v12[1].m128i_i64;
    do
    {
      if ( v13 )
      {
        v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
        v19 = v15[-3].m128i_i64[0];
        if ( v19 == m128i_i64 )
        {
          v13[1] = _mm_loadu_si128(v15 - 2);
        }
        else
        {
          v13->m128i_i64[0] = v19;
          v13[1].m128i_i64[0] = v15[-2].m128i_i64[0];
        }
        v13->m128i_i64[1] = v15[-3].m128i_i64[1];
        v15[-3].m128i_i64[0] = m128i_i64;
        v15[-3].m128i_i64[1] = 0;
        v15[-2].m128i_i8[0] = 0;
        v13[2].m128i_i64[0] = (__int64)v13[3].m128i_i64;
        v17 = (const __m128i *)v15[-1].m128i_i64[0];
        if ( v17 == v15 )
        {
          v13[3] = _mm_loadu_si128(v15);
        }
        else
        {
          v13[2].m128i_i64[0] = (__int64)v17;
          v13[3].m128i_i64[0] = v15->m128i_i64[0];
        }
        v13[2].m128i_i64[1] = v15[-1].m128i_i64[1];
        v18 = _mm_loadu_si128(v15 + 1);
        v15[-1].m128i_i64[0] = (__int64)v15;
        v15[-1].m128i_i64[1] = 0;
        v15->m128i_i8[0] = 0;
        v13[4] = v18;
      }
      v15 += 5;
      v13 += 5;
      m128i_i64 += 80;
    }
    while ( &v12[8].m128i_i8[((char *)v14 - (char *)v12 - 80) & 0xFFFFFFFFFFFFFFF0LL] != (__int8 *)v15 );
  }
  *(_DWORD *)(a1 + 8) = v60;
  v20 = *(unsigned __int64 **)a2;
  v21 = (unsigned __int64 *)(*(_QWORD *)a2 + 80LL * *(unsigned int *)(a2 + 8));
  if ( *(unsigned __int64 **)a2 != v21 )
  {
    do
    {
      v21 -= 10;
      v22 = v21[4];
      if ( (unsigned __int64 *)v22 != v21 + 6 )
        j_j___libc_free_0(v22);
      if ( (unsigned __int64 *)*v21 != v21 + 2 )
        j_j___libc_free_0(*v21);
    }
    while ( v20 != v21 );
  }
LABEL_23:
  *(_DWORD *)(a2 + 8) = 0;
}
