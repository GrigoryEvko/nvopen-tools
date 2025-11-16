// Function: sub_261A2E0
// Address: 0x261a2e0
//
void __fastcall sub_261A2E0(unsigned __int64 *a1, const __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 v6; // r12
  unsigned __int64 v7; // rbx
  const __m128i *v8; // r13
  const __m128i *v9; // r15
  __int64 v10; // rbx
  const __m128i *v11; // rdx
  __m128i *v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // rax
  bool v17; // cf
  unsigned __int64 v18; // rbx
  const __m128i *v19; // rcx
  __m128i *v20; // rax
  __m128i *v21; // rbx
  const __m128i *v22; // r12
  __m128i *v23; // r13
  unsigned __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 *v26; // rdx
  unsigned __int64 v27; // rdx
  __m128i *v28; // rax
  const __m128i *v29; // r12
  __m128i *v30; // r13
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // [rsp+8h] [rbp-68h]
  __int64 v35; // [rsp+10h] [rbp-60h]
  unsigned __int64 v37; // [rsp+18h] [rbp-58h]
  unsigned __int64 v38; // [rsp+18h] [rbp-58h]
  __int64 v39; // [rsp+20h] [rbp-50h]
  unsigned __int64 v40; // [rsp+20h] [rbp-50h]
  __int64 *v41; // [rsp+20h] [rbp-50h]
  unsigned __int64 v42; // [rsp+20h] [rbp-50h]
  __int64 v44; // [rsp+38h] [rbp-38h]

  if ( a3 == a4 )
    return;
  v5 = a3;
  v6 = a3;
  v7 = 0;
  do
  {
    ++v7;
    v6 = sub_220EF30(v6);
  }
  while ( a4 != v6 );
  v8 = (const __m128i *)a1[1];
  v9 = a2;
  if ( v7 <= (__int64)(a1[2] - (_QWORD)v8) >> 4 )
  {
    v35 = (char *)v8 - (char *)a2;
    v34 = v8 - a2;
    if ( v7 < v34 )
    {
      v10 = 16 * v7;
      if ( v8 == &v8[v10 / 0xFFFFFFFFFFFFFFF0LL] )
      {
        v13 = a1[1];
      }
      else
      {
        v11 = &v8[v10 / 0xFFFFFFFFFFFFFFF0LL];
        v12 = (__m128i *)a1[1];
        do
        {
          if ( v12 )
            *v12 = _mm_loadu_si128(v11);
          ++v12;
          ++v11;
        }
        while ( v12 != &v8[(unsigned __int64)v10 / 0x10] );
        v13 = a1[1];
      }
      a1[1] = v13 + v10;
      if ( a2 != &v8[v10 / 0xFFFFFFFFFFFFFFF0LL] )
        memmove((void *)&a2[(unsigned __int64)v10 / 0x10], a2, (char *)&v8[v10 / 0xFFFFFFFFFFFFFFF0LL] - (char *)a2);
      do
      {
        v14 = *(_QWORD *)(v5 + 40);
        ++v9;
        v9[-1].m128i_i64[0] = *(_QWORD *)(v5 + 32);
        v9[-1].m128i_i64[1] = v14;
        v5 = sub_220EF30(v5);
      }
      while ( v6 != v5 );
      return;
    }
    if ( v35 <= 0 )
    {
      v32 = v34 + 1;
      v44 = a3;
      if ( !v35 )
        goto LABEL_42;
      do
      {
        v42 = v32;
        v44 = sub_220EFE0(v44);
        v32 = v42 + 1;
      }
      while ( v42 );
    }
    else
    {
      v24 = v34 - 1;
      v44 = a3;
      do
      {
        v40 = v24;
        v44 = sub_220EF30(v44);
        v24 = v40 - 1;
      }
      while ( v40 );
    }
    if ( v6 == v44 )
    {
      v27 = (unsigned __int64)v8;
LABEL_47:
      v28 = (__m128i *)(v27 + 16 * (v7 - v34));
      a1[1] = (unsigned __int64)v28;
      if ( a2 != v8 )
      {
        v29 = a2;
        v30 = (__m128i *)((char *)v28 + (char *)v8 - (char *)a2);
        do
        {
          if ( v28 )
            *v28 = _mm_loadu_si128(v29);
          ++v28;
          ++v29;
        }
        while ( v28 != v30 );
        v28 = (__m128i *)a1[1];
      }
      a1[1] = (unsigned __int64)v28->m128i_u64 + v35;
      if ( a3 != v44 )
      {
        do
        {
          v31 = *(_QWORD *)(v5 + 40);
          ++v9;
          v9[-1].m128i_i64[0] = *(_QWORD *)(v5 + 32);
          v9[-1].m128i_i64[1] = v31;
          v5 = sub_220EF30(v5);
        }
        while ( v5 != v44 );
      }
      return;
    }
LABEL_42:
    v25 = v44;
    v26 = (__int64 *)v8;
    do
    {
      if ( v26 )
      {
        *v26 = *(_QWORD *)(v25 + 32);
        v26[1] = *(_QWORD *)(v25 + 40);
      }
      v41 = v26;
      v25 = sub_220EF30(v25);
      v26 = v41 + 2;
    }
    while ( v6 != v25 );
    v27 = a1[1];
    goto LABEL_47;
  }
  v15 = *a1;
  v16 = (__int64)((__int64)v8->m128i_i64 - *a1) >> 4;
  if ( v7 > 0x7FFFFFFFFFFFFFFLL - v16 )
    sub_4262D8((__int64)"vector::_M_range_insert");
  if ( v7 < v16 )
    v7 = (__int64)((__int64)v8->m128i_i64 - *a1) >> 4;
  v17 = __CFADD__(v16, v7);
  v18 = v16 + v7;
  if ( v17 )
  {
    v33 = 0x7FFFFFFFFFFFFFF0LL;
LABEL_66:
    v38 = v33;
    v39 = sub_22077B0(v33);
    v15 = *a1;
    v8 = (const __m128i *)a1[1];
    v37 = v39 + v38;
    goto LABEL_22;
  }
  if ( v18 )
  {
    if ( v18 > 0x7FFFFFFFFFFFFFFLL )
      v18 = 0x7FFFFFFFFFFFFFFLL;
    v33 = 16 * v18;
    goto LABEL_66;
  }
  v37 = 0;
  v39 = 0;
LABEL_22:
  if ( a2 == (const __m128i *)v15 )
  {
    v21 = (__m128i *)v39;
  }
  else
  {
    v19 = (const __m128i *)v15;
    v20 = (__m128i *)v39;
    v21 = (__m128i *)((char *)a2 + v39 - v15);
    do
    {
      if ( v20 )
        *v20 = _mm_loadu_si128(v19);
      ++v20;
      ++v19;
    }
    while ( v20 != v21 );
  }
  do
  {
    if ( v21 )
    {
      v21->m128i_i64[0] = *(_QWORD *)(v5 + 32);
      v21->m128i_i64[1] = *(_QWORD *)(v5 + 40);
    }
    ++v21;
    v5 = sub_220EF30(v5);
  }
  while ( v6 != v5 );
  v22 = a2;
  if ( a2 == v8 )
  {
    v23 = v21;
  }
  else
  {
    v23 = (__m128i *)((char *)v21 + (char *)v8 - (char *)a2);
    do
    {
      if ( v21 )
        *v21 = _mm_loadu_si128(v22);
      ++v21;
      ++v22;
    }
    while ( v23 != v21 );
  }
  if ( v15 )
    j_j___libc_free_0(v15);
  *a1 = v39;
  a1[1] = (unsigned __int64)v23;
  a1[2] = v37;
}
