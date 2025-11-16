// Function: sub_9CC750
// Address: 0x9cc750
//
__m128i **__fastcall sub_9CC750(__m128i **a1, const __m128i *a2, __int64 a3, _QWORD *a4)
{
  const __m128i *v6; // r12
  const __m128i *v7; // r14
  __int64 v8; // rax
  __int64 v9; // rcx
  bool v10; // cf
  unsigned __int64 v11; // rax
  signed __int64 v12; // r9
  __int64 m128i_i64; // r15
  _BYTE *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // r8
  const __m128i *v20; // rax
  __m128i *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rsi
  __m128i *v24; // r15
  const __m128i *i; // r13
  const __m128i *v26; // rdx
  __int64 v27; // rdx
  const __m128i *v28; // rdi
  __int64 v29; // rdi
  __int64 v31; // r15
  __int64 v32; // rax
  _QWORD *v33; // [rsp+0h] [rbp-80h]
  __int64 *v34; // [rsp+8h] [rbp-78h]
  __int64 v35; // [rsp+8h] [rbp-78h]
  __m128i *v36; // [rsp+10h] [rbp-70h]
  __int64 v37; // [rsp+18h] [rbp-68h]
  __m128i *v39; // [rsp+28h] [rbp-58h]
  __int64 v40[2]; // [rsp+30h] [rbp-50h] BYREF
  __m128i v41[4]; // [rsp+40h] [rbp-40h] BYREF

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
    v31 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v37 = 0;
      m128i_i64 = 56;
      v39 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x249249249249249LL )
      v11 = 0x249249249249249LL;
    v31 = 56 * v11;
  }
  v33 = a4;
  v35 = a3;
  v32 = sub_22077B0(v31);
  v12 = (char *)a2 - (char *)v7;
  a3 = v35;
  v39 = (__m128i *)v32;
  a4 = v33;
  v37 = v32 + v31;
  m128i_i64 = v32 + 56;
LABEL_7:
  v14 = *(_BYTE **)a3;
  v34 = a4;
  v36 = (__m128i *)((char *)v39 + v12);
  v15 = *(_QWORD *)a3 + *(_QWORD *)(a3 + 8);
  v40[0] = (__int64)v41;
  sub_9C36C0(v40, v14, v15);
  v16 = *v34;
  v17 = v34[1];
  *v34 = 0;
  v18 = v34[2];
  v34[1] = 0;
  v34[2] = 0;
  if ( v36 )
  {
    v36->m128i_i64[0] = (__int64)v36[1].m128i_i64;
    if ( (__m128i *)v40[0] == v41 )
    {
      v36[1] = _mm_load_si128(v41);
    }
    else
    {
      v36->m128i_i64[0] = v40[0];
      v36[1].m128i_i64[0] = v41[0].m128i_i64[0];
    }
    v19 = v40[1];
    v36[2].m128i_i64[0] = v16;
    v36[2].m128i_i64[1] = v17;
    v36->m128i_i64[1] = v19;
    v36[3].m128i_i64[0] = v18;
    if ( a2 == v7 )
      goto LABEL_11;
    goto LABEL_22;
  }
  v23 = v18 - v16;
  if ( v16 )
    j_j___libc_free_0(v16, v23);
  if ( (__m128i *)v40[0] != v41 )
    j_j___libc_free_0(v40[0], v41[0].m128i_i64[0] + 1);
  if ( a2 != v7 )
  {
LABEL_22:
    v24 = v39;
    for ( i = v7 + 1; ; i = (const __m128i *)((char *)i + 56) )
    {
      if ( v24 )
      {
        v24->m128i_i64[0] = (__int64)v24[1].m128i_i64;
        v26 = (const __m128i *)i[-1].m128i_i64[0];
        if ( i == v26 )
        {
          v24[1] = _mm_loadu_si128(i);
        }
        else
        {
          v24->m128i_i64[0] = (__int64)v26;
          v24[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v24->m128i_i64[1] = i[-1].m128i_i64[1];
        v27 = i[1].m128i_i64[0];
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v24[2].m128i_i64[0] = v27;
        v24[2].m128i_i64[1] = i[1].m128i_i64[1];
        v24[3].m128i_i64[0] = i[2].m128i_i64[0];
        i[2].m128i_i64[0] = 0;
        i[1].m128i_i64[0] = 0;
      }
      else
      {
        v29 = i[1].m128i_i64[0];
        if ( v29 )
          j_j___libc_free_0(v29, i[2].m128i_i64[0] - v29);
      }
      v28 = (const __m128i *)i[-1].m128i_i64[0];
      if ( i != v28 )
        j_j___libc_free_0(v28, i->m128i_i64[0] + 1);
      if ( a2 == (const __m128i *)&i[2].m128i_u64[1] )
        break;
      v24 = (__m128i *)((char *)v24 + 56);
    }
    m128i_i64 = (__int64)v24[7].m128i_i64;
  }
LABEL_11:
  if ( a2 != v6 )
  {
    v20 = a2;
    v21 = (__m128i *)m128i_i64;
    do
    {
      v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
      if ( (const __m128i *)v20->m128i_i64[0] == &v20[1] )
      {
        v21[1] = _mm_loadu_si128(v20 + 1);
      }
      else
      {
        v21->m128i_i64[0] = v20->m128i_i64[0];
        v21[1].m128i_i64[0] = v20[1].m128i_i64[0];
      }
      v22 = v20->m128i_i64[1];
      v20 = (const __m128i *)((char *)v20 + 56);
      v21 = (__m128i *)((char *)v21 + 56);
      v21[-3].m128i_i64[0] = v22;
      v21[-2].m128i_i64[1] = v20[-2].m128i_i64[1];
      v21[-1].m128i_i64[0] = v20[-1].m128i_i64[0];
      v21[-1].m128i_i64[1] = v20[-1].m128i_i64[1];
    }
    while ( v20 != v6 );
    m128i_i64 += 56
               * (((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)((char *)v20 - (char *)a2 - 56) >> 3))
                 & 0x1FFFFFFFFFFFFFFFLL)
                + 1);
  }
  if ( v7 )
    j_j___libc_free_0(v7, (char *)a1[2] - (char *)v7);
  *a1 = v39;
  a1[1] = (__m128i *)m128i_i64;
  a1[2] = (__m128i *)v37;
  return a1;
}
