// Function: sub_35914E0
// Address: 0x35914e0
//
unsigned __int64 *__fastcall sub_35914E0(unsigned __int64 *a1, const __m128i *a2, __int64 a3)
{
  unsigned __int64 v4; // r15
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rcx
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int8 *v10; // rcx
  __int64 v11; // rbx
  char *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned __int64 v15; // r12
  const __m128i *i; // rbx
  const __m128i *v17; // rdx
  unsigned __int64 v18; // rdi
  __int64 v19; // rdx
  const __m128i *v20; // rax
  __int64 v21; // rcx
  const __m128i *v22; // rcx
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-58h]
  unsigned __int64 v27; // [rsp+18h] [rbp-48h]
  unsigned __int64 v29; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - *a1) >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - v5) >> 3);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x3333333333333333LL * ((__int64)(v4 - v5) >> 3);
  v10 = &a2->m128i_i8[-v5];
  if ( v8 )
  {
    v24 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v27 = 0;
      v11 = 40;
      v29 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x333333333333333LL )
      v9 = 0x333333333333333LL;
    v24 = 40 * v9;
  }
  v26 = a3;
  v25 = sub_22077B0(v24);
  v10 = &a2->m128i_i8[-v5];
  a3 = v26;
  v29 = v25;
  v27 = v25 + v24;
  v11 = v25 + 40;
LABEL_7:
  v12 = &v10[v29];
  if ( &v10[v29] )
  {
    v13 = *(_QWORD *)(a3 + 8);
    *(_DWORD *)v12 = *(_DWORD *)a3;
    *((_QWORD *)v12 + 1) = v12 + 24;
    if ( v13 == a3 + 24 )
    {
      *(__m128i *)(v12 + 24) = _mm_loadu_si128((const __m128i *)(a3 + 24));
    }
    else
    {
      *((_QWORD *)v12 + 1) = v13;
      *((_QWORD *)v12 + 3) = *(_QWORD *)(a3 + 24);
    }
    v14 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a3 + 8) = a3 + 24;
    *(_QWORD *)(a3 + 16) = 0;
    *((_QWORD *)v12 + 2) = v14;
    *(_BYTE *)(a3 + 24) = 0;
  }
  if ( a2 != (const __m128i *)v5 )
  {
    v15 = v29;
    for ( i = (const __m128i *)(v5 + 24); ; i = (const __m128i *)((char *)i + 40) )
    {
      if ( v15 )
      {
        *(_DWORD *)v15 = i[-2].m128i_i32[2];
        *(_QWORD *)(v15 + 8) = v15 + 24;
        v17 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v17 == i )
        {
          *(__m128i *)(v15 + 24) = _mm_loadu_si128(i);
        }
        else
        {
          *(_QWORD *)(v15 + 8) = v17;
          *(_QWORD *)(v15 + 24) = i->m128i_i64[0];
        }
        *(_QWORD *)(v15 + 16) = i[-1].m128i_i64[1];
        i[-1].m128i_i64[0] = (__int64)i;
      }
      else
      {
        v18 = i[-1].m128i_u64[0];
        if ( (const __m128i *)v18 != i )
          j_j___libc_free_0(v18);
      }
      if ( a2 == &i[1] )
        break;
      v15 += 40LL;
    }
    v11 = v15 + 80;
  }
  if ( a2 != (const __m128i *)v4 )
  {
    v19 = v11;
    v20 = a2;
    do
    {
      *(_DWORD *)v19 = v20->m128i_i32[0];
      *(_QWORD *)(v19 + 8) = v19 + 24;
      v22 = (const __m128i *)v20->m128i_i64[1];
      if ( v22 == (const __m128i *)&v20[1].m128i_u64[1] )
      {
        *(__m128i *)(v19 + 24) = _mm_loadu_si128((const __m128i *)((char *)v20 + 24));
      }
      else
      {
        *(_QWORD *)(v19 + 8) = v22;
        *(_QWORD *)(v19 + 24) = v20[1].m128i_i64[1];
      }
      v21 = v20[1].m128i_i64[0];
      v20 = (const __m128i *)((char *)v20 + 40);
      v19 += 40;
      *(_QWORD *)(v19 - 24) = v21;
    }
    while ( (const __m128i *)v4 != v20 );
    v11 += 8 * ((v4 - (unsigned __int64)a2 - 40) >> 3) + 40;
  }
  if ( v5 )
    j_j___libc_free_0(v5);
  *a1 = v29;
  a1[1] = v11;
  a1[2] = v27;
  return a1;
}
