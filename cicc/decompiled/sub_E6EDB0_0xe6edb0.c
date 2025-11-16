// Function: sub_E6EDB0
// Address: 0xe6edb0
//
void __fastcall sub_E6EDB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // r13
  unsigned __int64 v8; // rbx
  __int64 v9; // r8
  unsigned __int64 v10; // r9
  const __m128i *v11; // r8
  __m128i *v12; // rdx
  const __m128i *v13; // r14
  const __m128i *v14; // rax
  __m128i *v15; // rsi
  __m128i v16; // xmm0
  const __m128i *v17; // rcx
  __int64 v18; // r13
  __int64 v19; // rbx
  __int64 v20; // rdi
  unsigned __int64 v21; // rbx
  __int64 v22; // rdi
  __int64 v23; // rax
  unsigned __int64 v24; // rbx
  __int64 v25; // rdi
  __int64 v26; // r13
  __int64 v27; // rbx
  __int64 v28; // rdi
  __m128i *v29; // r14
  __int64 v30; // r8
  __m128i *v31; // rcx
  __m128i *v32; // r13
  __int64 v33; // rdx
  __m128i *v34; // rdi
  __m128i *v35; // rax
  size_t v36; // rdx
  unsigned __int64 v37; // rbx
  __int64 v38; // rdi
  __m128i *v39; // rbx
  __int64 v40; // r8
  __m128i *v41; // r14
  __int64 v42; // r10
  __int64 v43; // rax
  __m128i *v44; // rdi
  __m128i *v45; // r9
  size_t v46; // rdx
  __int64 v47; // [rsp-58h] [rbp-58h]
  __int64 v48; // [rsp-50h] [rbp-50h]
  __m128i *v49; // [rsp-50h] [rbp-50h]
  int v50; // [rsp-44h] [rbp-44h]
  __int64 v51; // [rsp-40h] [rbp-40h]
  __m128i *v52; // [rsp-40h] [rbp-40h]
  unsigned __int64 v53; // [rsp-40h] [rbp-40h]
  __int64 v54; // [rsp-40h] [rbp-40h]

  if ( a1 == a2 )
    return;
  v5 = a2 + 16;
  v6 = a2;
  v7 = *(_QWORD *)a1;
  v8 = *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1;
  if ( *(_QWORD *)a2 != a2 + 16 )
  {
    v21 = v7 + 48 * v8;
    if ( v21 != v7 )
    {
      do
      {
        v21 -= 48LL;
        v22 = *(_QWORD *)(v21 + 16);
        if ( v22 != v21 + 32 )
        {
          a2 = *(_QWORD *)(v21 + 32) + 1LL;
          j_j___libc_free_0(v22, a2);
        }
      }
      while ( v21 != v7 );
      v9 = *(_QWORD *)a1;
    }
    if ( v9 != a1 + 16 )
      _libc_free(v9, a2);
    *(_QWORD *)a1 = *(_QWORD *)v6;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(v6 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(v6 + 12);
    *(_QWORD *)v6 = v5;
    *(_QWORD *)(v6 + 8) = 0;
    return;
  }
  v10 = *(unsigned int *)(a2 + 8);
  v50 = *(_DWORD *)(a2 + 8);
  if ( v10 <= v8 )
  {
    v23 = *(_QWORD *)a1;
    if ( !*(_DWORD *)(a2 + 8) )
    {
LABEL_28:
      v24 = v23 + 48 * v8;
      while ( v9 != v24 )
      {
        v24 -= 48LL;
        v25 = *(_QWORD *)(v24 + 16);
        if ( v25 != v24 + 32 )
        {
          v51 = v9;
          j_j___libc_free_0(v25, *(_QWORD *)(v24 + 32) + 1LL);
          v9 = v51;
        }
      }
      *(_DWORD *)(a1 + 8) = v50;
      v26 = *(_QWORD *)a2;
      v27 = *(_QWORD *)a2 + 48LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v27 )
      {
        do
        {
          v27 -= 48;
          v28 = *(_QWORD *)(v27 + 16);
          if ( v28 != v27 + 32 )
            j_j___libc_free_0(v28, *(_QWORD *)(v27 + 32) + 1LL);
        }
        while ( v26 != v27 );
      }
      goto LABEL_18;
    }
    v39 = (__m128i *)(a2 + 48);
    v40 = 48 * v10;
    v41 = (__m128i *)(v7 + 32);
    v42 = a2 + 48 + 48 * v10;
    while ( 1 )
    {
      v44 = (__m128i *)v41[-1].m128i_i64[0];
      v41[-2] = _mm_loadu_si128(v39 - 2);
      v45 = (__m128i *)v39[-1].m128i_i64[0];
      if ( v39 == v45 )
      {
        v46 = v39[-1].m128i_u64[1];
        if ( v46 )
        {
          if ( v46 == 1 )
          {
            v44->m128i_i8[0] = v39->m128i_i8[0];
            v46 = v39[-1].m128i_u64[1];
            v44 = (__m128i *)v41[-1].m128i_i64[0];
          }
          else
          {
            v47 = v40;
            v49 = (__m128i *)v39[-1].m128i_i64[0];
            v54 = v42;
            memcpy(v44, v39, v46);
            v46 = v39[-1].m128i_u64[1];
            v44 = (__m128i *)v41[-1].m128i_i64[0];
            v40 = v47;
            v45 = v49;
            v42 = v54;
          }
        }
        v41[-1].m128i_i64[1] = v46;
        v44->m128i_i8[v46] = 0;
        v44 = (__m128i *)v45[-1].m128i_i64[0];
        goto LABEL_60;
      }
      if ( v44 == v41 )
        break;
      v41[-1].m128i_i64[0] = (__int64)v45;
      v43 = v41->m128i_i64[0];
      v41[-1].m128i_i64[1] = v39[-1].m128i_i64[1];
      v41->m128i_i64[0] = v39->m128i_i64[0];
      if ( !v44 )
        goto LABEL_67;
      v39[-1].m128i_i64[0] = (__int64)v44;
      v39->m128i_i64[0] = v43;
LABEL_60:
      v39[-1].m128i_i64[1] = 0;
      v39 += 3;
      v41 += 3;
      v44->m128i_i8[0] = 0;
      if ( (__m128i *)v42 == v39 )
      {
        v23 = *(_QWORD *)a1;
        v8 = *(unsigned int *)(a1 + 8);
        v9 = v7 + v40;
        goto LABEL_28;
      }
    }
    v41[-1].m128i_i64[0] = (__int64)v45;
    v41[-1].m128i_i64[1] = v39[-1].m128i_i64[1];
    v41->m128i_i64[0] = v39->m128i_i64[0];
LABEL_67:
    v39[-1].m128i_i64[0] = (__int64)v39;
    v44 = v39;
    goto LABEL_60;
  }
  if ( v10 > *(unsigned int *)(a1 + 12) )
  {
    v37 = v7 + 48 * v8;
    while ( v37 != v7 )
    {
      while ( 1 )
      {
        v37 -= 48LL;
        v38 = *(_QWORD *)(v37 + 16);
        if ( v38 == v37 + 32 )
          break;
        v53 = v10;
        j_j___libc_free_0(v38, *(_QWORD *)(v37 + 32) + 1LL);
        v10 = v53;
        if ( v37 == v7 )
          goto LABEL_54;
      }
    }
LABEL_54:
    *(_DWORD *)(a1 + 8) = 0;
    v8 = 0;
    sub_C8F9C0(a1, v10, a3, a4, v9, v10);
    v5 = *(_QWORD *)a2;
    v10 = *(unsigned int *)(a2 + 8);
    v7 = *(_QWORD *)a1;
    v11 = *(const __m128i **)a2;
    goto LABEL_6;
  }
  v11 = (const __m128i *)(a2 + 16);
  if ( !*(_DWORD *)(a1 + 8) )
    goto LABEL_6;
  v29 = (__m128i *)(a2 + 48);
  v30 = 48 * v8;
  v31 = (__m128i *)(v7 + 32);
  v8 = v30;
  v32 = (__m128i *)(a2 + 48 + v30);
  do
  {
    v34 = (__m128i *)v31[-1].m128i_i64[0];
    v31[-2] = _mm_loadu_si128(v29 - 2);
    v35 = (__m128i *)v29[-1].m128i_i64[0];
    if ( v35 == v29 )
    {
      v36 = v29[-1].m128i_u64[1];
      if ( v36 )
      {
        if ( v36 == 1 )
        {
          v34->m128i_i8[0] = v29->m128i_i8[0];
          v36 = v29[-1].m128i_u64[1];
          v34 = (__m128i *)v31[-1].m128i_i64[0];
        }
        else
        {
          v48 = v30;
          v52 = v31;
          memcpy(v34, v29, v36);
          v31 = v52;
          v36 = v29[-1].m128i_u64[1];
          v30 = v48;
          v34 = (__m128i *)v52[-1].m128i_i64[0];
        }
      }
      v31[-1].m128i_i64[1] = v36;
      v34->m128i_i8[v36] = 0;
      v34 = (__m128i *)v29[-1].m128i_i64[0];
    }
    else
    {
      if ( v34 == v31 )
      {
        v31[-1].m128i_i64[0] = (__int64)v35;
        v31[-1].m128i_i64[1] = v29[-1].m128i_i64[1];
        v31->m128i_i64[0] = v29->m128i_i64[0];
      }
      else
      {
        v31[-1].m128i_i64[0] = (__int64)v35;
        v33 = v31->m128i_i64[0];
        v31[-1].m128i_i64[1] = v29[-1].m128i_i64[1];
        v31->m128i_i64[0] = v29->m128i_i64[0];
        if ( v34 )
        {
          v29[-1].m128i_i64[0] = (__int64)v34;
          v29->m128i_i64[0] = v33;
          goto LABEL_41;
        }
      }
      v29[-1].m128i_i64[0] = (__int64)v29;
      v34 = v29;
    }
LABEL_41:
    v29[-1].m128i_i64[1] = 0;
    v29 += 3;
    v31 += 3;
    v34->m128i_i8[0] = 0;
  }
  while ( v29 != v32 );
  v5 = *(_QWORD *)a2;
  v10 = *(unsigned int *)(a2 + 8);
  v7 = *(_QWORD *)a1;
  v11 = (const __m128i *)(*(_QWORD *)a2 + v30);
LABEL_6:
  v12 = (__m128i *)(v7 + v8);
  v13 = (const __m128i *)(48 * v10 + v5);
  if ( v13 != v11 )
  {
    v14 = v11 + 2;
    v15 = &v12[3
             * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v13 - (char *)v11 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
             + 3];
    do
    {
      if ( v12 )
      {
        v16 = _mm_loadu_si128(v14 - 2);
        v12[1].m128i_i64[0] = (__int64)v12[2].m128i_i64;
        *v12 = v16;
        v17 = (const __m128i *)v14[-1].m128i_i64[0];
        if ( v17 == v14 )
        {
          v12[2] = _mm_loadu_si128(v14);
        }
        else
        {
          v12[1].m128i_i64[0] = (__int64)v17;
          v12[2].m128i_i64[0] = v14->m128i_i64[0];
        }
        v12[1].m128i_i64[1] = v14[-1].m128i_i64[1];
        v14[-1].m128i_i64[0] = (__int64)v14;
        v14[-1].m128i_i64[1] = 0;
        v14->m128i_i8[0] = 0;
      }
      v12 += 3;
      v14 += 3;
    }
    while ( v12 != v15 );
  }
  *(_DWORD *)(a1 + 8) = v50;
  v18 = *(_QWORD *)v6;
  v19 = *(_QWORD *)v6 + 48LL * *(unsigned int *)(v6 + 8);
  if ( *(_QWORD *)v6 != v19 )
  {
    do
    {
      v19 -= 48;
      v20 = *(_QWORD *)(v19 + 16);
      if ( v20 != v19 + 32 )
        j_j___libc_free_0(v20, *(_QWORD *)(v19 + 32) + 1LL);
    }
    while ( v18 != v19 );
  }
LABEL_18:
  *(_DWORD *)(v6 + 8) = 0;
}
