// Function: sub_E9C930
// Address: 0xe9c930
//
__int64 *__fastcall sub_E9C930(__int64 *a1, const __m128i *a2, __int64 a3)
{
  const __m128i *v4; // r12
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rcx
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int8 *v10; // rcx
  __int64 v11; // r8
  char *v12; // rax
  __m128i v13; // xmm4
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // r14
  const __m128i *i; // r13
  const __m128i *v22; // rcx
  __int64 v23; // rdi
  const __m128i *v24; // rdi
  const __m128i *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __m128i v28; // xmm1
  __int64 v29; // rcx
  __int64 v30; // rcx
  __int64 v31; // rcx
  __int64 v32; // rcx
  const __m128i *v33; // rcx
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+8h] [rbp-58h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+28h] [rbp-38h]

  v4 = (const __m128i *)a1[1];
  v5 = *a1;
  v6 = 0x4EC4EC4EC4EC4EC5LL * (((__int64)v4->m128i_i64 - *a1) >> 3);
  if ( v6 == 0x13B13B13B13B13BLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0x4EC4EC4EC4EC4EC5LL * (((__int64)v4->m128i_i64 - v5) >> 3);
  v8 = __CFADD__(v7, v6);
  v9 = v7 + v6;
  v10 = &a2->m128i_i8[-v5];
  if ( v8 )
  {
    v35 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v39 = 0;
      v11 = 104;
      v41 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x13B13B13B13B13BLL )
      v9 = 0x13B13B13B13B13BLL;
    v35 = 104 * v9;
  }
  v38 = a3;
  v36 = sub_22077B0(v35);
  v10 = &a2->m128i_i8[-v5];
  v41 = v36;
  a3 = v38;
  v11 = v36 + 104;
  v39 = v36 + v35;
LABEL_7:
  v12 = &v10[v41];
  if ( &v10[v41] )
  {
    v13 = _mm_loadu_si128((const __m128i *)(a3 + 8));
    v14 = *(_QWORD *)(a3 + 72);
    *(_QWORD *)v12 = *(_QWORD *)a3;
    v15 = *(_QWORD *)(a3 + 24);
    *(__m128i *)(v12 + 8) = v13;
    *((_QWORD *)v12 + 3) = v15;
    v12[32] = *(_BYTE *)(a3 + 32);
    *((_QWORD *)v12 + 5) = *(_QWORD *)(a3 + 40);
    v16 = *(_QWORD *)(a3 + 48);
    *(_QWORD *)(a3 + 48) = 0;
    *((_QWORD *)v12 + 6) = v16;
    v17 = *(_QWORD *)(a3 + 56);
    *(_QWORD *)(a3 + 56) = 0;
    *((_QWORD *)v12 + 7) = v17;
    v18 = *(_QWORD *)(a3 + 64);
    *(_QWORD *)(a3 + 64) = 0;
    *((_QWORD *)v12 + 8) = v18;
    *((_QWORD *)v12 + 9) = v12 + 88;
    if ( v14 == a3 + 88 )
    {
      *(__m128i *)(v12 + 88) = _mm_loadu_si128((const __m128i *)(a3 + 88));
    }
    else
    {
      *((_QWORD *)v12 + 9) = v14;
      *((_QWORD *)v12 + 11) = *(_QWORD *)(a3 + 88);
    }
    v19 = *(_QWORD *)(a3 + 80);
    *(_QWORD *)(a3 + 72) = a3 + 88;
    *(_QWORD *)(a3 + 80) = 0;
    *((_QWORD *)v12 + 10) = v19;
    *(_BYTE *)(a3 + 88) = 0;
  }
  if ( a2 != (const __m128i *)v5 )
  {
    v20 = v41;
    for ( i = (const __m128i *)(v5 + 88); ; i = (const __m128i *)((char *)i + 104) )
    {
      if ( v20 )
      {
        *(_QWORD *)v20 = i[-6].m128i_i64[1];
        *(__m128i *)(v20 + 8) = _mm_loadu_si128(i - 5);
        *(_QWORD *)(v20 + 24) = i[-4].m128i_i64[0];
        *(_BYTE *)(v20 + 32) = i[-4].m128i_i8[8];
        *(_QWORD *)(v20 + 40) = i[-3].m128i_i64[0];
        *(_QWORD *)(v20 + 48) = i[-3].m128i_i64[1];
        *(_QWORD *)(v20 + 56) = i[-2].m128i_i64[0];
        *(_QWORD *)(v20 + 64) = i[-2].m128i_i64[1];
        i[-2].m128i_i64[1] = 0;
        i[-2].m128i_i64[0] = 0;
        i[-3].m128i_i64[1] = 0;
        *(_QWORD *)(v20 + 72) = v20 + 88;
        v22 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v22 == i )
        {
          *(__m128i *)(v20 + 88) = _mm_loadu_si128(i);
        }
        else
        {
          *(_QWORD *)(v20 + 72) = v22;
          *(_QWORD *)(v20 + 88) = i->m128i_i64[0];
        }
        *(_QWORD *)(v20 + 80) = i[-1].m128i_i64[1];
        i[-1].m128i_i64[0] = (__int64)i;
      }
      else
      {
        v24 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v24 != i )
          j_j___libc_free_0(v24, i->m128i_i64[0] + 1);
      }
      v23 = i[-3].m128i_i64[1];
      if ( v23 )
        j_j___libc_free_0(v23, i[-2].m128i_i64[1] - v23);
      if ( a2 == &i[1] )
        break;
      v20 += 104;
    }
    v11 = v20 + 208;
  }
  if ( a2 != v4 )
  {
    v25 = a2;
    v26 = v11;
    do
    {
      v28 = _mm_loadu_si128((const __m128i *)&v25->m128i_u64[1]);
      *(_QWORD *)v26 = v25->m128i_i64[0];
      v29 = v25[1].m128i_i64[1];
      *(__m128i *)(v26 + 8) = v28;
      *(_QWORD *)(v26 + 24) = v29;
      *(_BYTE *)(v26 + 32) = v25[2].m128i_i8[0];
      *(_QWORD *)(v26 + 40) = v25[2].m128i_i64[1];
      v30 = v25[3].m128i_i64[0];
      v25[3].m128i_i64[0] = 0;
      *(_QWORD *)(v26 + 48) = v30;
      v31 = v25[3].m128i_i64[1];
      v25[3].m128i_i64[1] = 0;
      *(_QWORD *)(v26 + 56) = v31;
      v32 = v25[4].m128i_i64[0];
      v25[4].m128i_i64[0] = 0;
      *(_QWORD *)(v26 + 64) = v32;
      *(_QWORD *)(v26 + 72) = v26 + 88;
      v33 = (const __m128i *)v25[4].m128i_i64[1];
      if ( v33 == (const __m128i *)&v25[5].m128i_u64[1] )
      {
        *(__m128i *)(v26 + 88) = _mm_loadu_si128((const __m128i *)((char *)v25 + 88));
      }
      else
      {
        *(_QWORD *)(v26 + 72) = v33;
        *(_QWORD *)(v26 + 88) = v25[5].m128i_i64[1];
      }
      v27 = v25[5].m128i_i64[0];
      v25 = (const __m128i *)((char *)v25 + 104);
      v26 += 104;
      *(_QWORD *)(v26 - 24) = v27;
    }
    while ( v25 != v4 );
    v11 += 104
         * (((0xEC4EC4EC4EC4EC5LL * ((unsigned __int64)((char *)v25 - (char *)a2 - 104) >> 3)) & 0x1FFFFFFFFFFFFFFFLL)
          + 1);
  }
  if ( v5 )
  {
    v37 = v11;
    j_j___libc_free_0(v5, a1[2] - v5);
    v11 = v37;
  }
  *a1 = v41;
  a1[1] = v11;
  a1[2] = v39;
  return a1;
}
