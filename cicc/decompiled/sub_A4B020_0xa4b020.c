// Function: sub_A4B020
// Address: 0xa4b020
//
__int64 *__fastcall sub_A4B020(__int64 *a1, const __m128i *a2, _DWORD *a3, __m128i *a4)
{
  const __m128i *v5; // r15
  __int64 v6; // r14
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rsi
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int8 *v11; // rsi
  __int64 v12; // rbx
  char *v13; // rax
  __m128i *v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // r12
  const __m128i *i; // rbx
  const __m128i *v18; // rdx
  const __m128i *v19; // rdi
  __int64 v20; // rdx
  const __m128i *v21; // rax
  __int64 v22; // rcx
  const __m128i *v23; // rcx
  __int64 v25; // rbx
  __int64 v26; // rax
  __m128i *v27; // [rsp+0h] [rbp-60h]
  _DWORD *v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+28h] [rbp-38h]

  v5 = (const __m128i *)a1[1];
  v6 = *a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v5->m128i_i64 - *a1) >> 3);
  if ( v7 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v5->m128i_i64 - v6) >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x3333333333333333LL * (((__int64)v5->m128i_i64 - v6) >> 3);
  v11 = &a2->m128i_i8[-v6];
  if ( v9 )
  {
    v25 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v29 = 0;
      v12 = 40;
      v31 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x333333333333333LL )
      v10 = 0x333333333333333LL;
    v25 = 40 * v10;
  }
  v27 = a4;
  v28 = a3;
  v26 = sub_22077B0(v25);
  v11 = &a2->m128i_i8[-v6];
  a3 = v28;
  v31 = v26;
  a4 = v27;
  v29 = v26 + v25;
  v12 = v26 + 40;
LABEL_7:
  v13 = &v11[v31];
  if ( &v11[v31] )
  {
    v14 = (__m128i *)a4->m128i_i64[0];
    *(_DWORD *)v13 = *a3;
    *((_QWORD *)v13 + 1) = v13 + 24;
    if ( v14 == &a4[1] )
    {
      *(__m128i *)(v13 + 24) = _mm_loadu_si128(a4 + 1);
    }
    else
    {
      *((_QWORD *)v13 + 1) = v14;
      *((_QWORD *)v13 + 3) = a4[1].m128i_i64[0];
    }
    v15 = a4->m128i_i64[1];
    a4->m128i_i64[0] = (__int64)a4[1].m128i_i64;
    a4->m128i_i64[1] = 0;
    *((_QWORD *)v13 + 2) = v15;
    a4[1].m128i_i8[0] = 0;
  }
  if ( a2 != (const __m128i *)v6 )
  {
    v16 = v31;
    for ( i = (const __m128i *)(v6 + 24); ; i = (const __m128i *)((char *)i + 40) )
    {
      if ( v16 )
      {
        *(_DWORD *)v16 = i[-2].m128i_i32[2];
        *(_QWORD *)(v16 + 8) = v16 + 24;
        v18 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v18 == i )
        {
          *(__m128i *)(v16 + 24) = _mm_loadu_si128(i);
        }
        else
        {
          *(_QWORD *)(v16 + 8) = v18;
          *(_QWORD *)(v16 + 24) = i->m128i_i64[0];
        }
        *(_QWORD *)(v16 + 16) = i[-1].m128i_i64[1];
        i[-1].m128i_i64[0] = (__int64)i;
      }
      else
      {
        v19 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v19 != i )
          j_j___libc_free_0(v19, i->m128i_i64[0] + 1);
      }
      if ( a2 == &i[1] )
        break;
      v16 += 40;
    }
    v12 = v16 + 80;
  }
  if ( a2 != v5 )
  {
    v20 = v12;
    v21 = a2;
    do
    {
      *(_DWORD *)v20 = v21->m128i_i32[0];
      *(_QWORD *)(v20 + 8) = v20 + 24;
      v23 = (const __m128i *)v21->m128i_i64[1];
      if ( v23 == (const __m128i *)&v21[1].m128i_u64[1] )
      {
        *(__m128i *)(v20 + 24) = _mm_loadu_si128((const __m128i *)((char *)v21 + 24));
      }
      else
      {
        *(_QWORD *)(v20 + 8) = v23;
        *(_QWORD *)(v20 + 24) = v21[1].m128i_i64[1];
      }
      v22 = v21[1].m128i_i64[0];
      v21 = (const __m128i *)((char *)v21 + 40);
      v20 += 40;
      *(_QWORD *)(v20 - 24) = v22;
    }
    while ( v5 != v21 );
    v12 += 8 * ((unsigned __int64)((char *)v5 - (char *)a2 - 40) >> 3) + 40;
  }
  if ( v6 )
    j_j___libc_free_0(v6, a1[2] - v6);
  *a1 = v31;
  a1[1] = v12;
  a1[2] = v29;
  return a1;
}
