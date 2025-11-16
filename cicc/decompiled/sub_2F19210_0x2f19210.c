// Function: sub_2F19210
// Address: 0x2f19210
//
unsigned __int64 *__fastcall sub_2F19210(unsigned __int64 *a1, const __m128i *a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rsi
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int8 *v10; // rsi
  __int64 v11; // r8
  char *v12; // rax
  __m128i v13; // xmm6
  __int64 v14; // rdi
  __int64 v15; // rdi
  __m128i v16; // xmm7
  __int16 v17; // si
  char v18; // dl
  unsigned __int64 v19; // r13
  const __m128i *i; // r12
  __m128i v21; // xmm2
  unsigned __int64 v22; // rdi
  const __m128i *v23; // rdx
  const __m128i *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  __m128i v27; // xmm0
  __m128i v28; // xmm3
  const __m128i *v29; // rsi
  unsigned __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  unsigned __int64 v35; // [rsp+18h] [rbp-48h]
  unsigned __int64 v37; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - *a1) >> 4);
  if ( v6 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - v5) >> 4);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x3333333333333333LL * ((__int64)(v4 - v5) >> 4);
  v10 = &a2->m128i_i8[-v5];
  if ( v8 )
  {
    v31 = 0x7FFFFFFFFFFFFFD0LL;
  }
  else
  {
    if ( !v9 )
    {
      v35 = 0;
      v11 = 80;
      v37 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x199999999999999LL )
      v9 = 0x199999999999999LL;
    v31 = 80 * v9;
  }
  v34 = a3;
  v32 = sub_22077B0(v31);
  v10 = &a2->m128i_i8[-v5];
  a3 = v34;
  v37 = v32;
  v11 = v32 + 80;
  v35 = v32 + v31;
LABEL_7:
  v12 = &v10[v37];
  if ( &v10[v37] )
  {
    v13 = _mm_loadu_si128((const __m128i *)a3);
    v14 = *(_QWORD *)(a3 + 24);
    *((_QWORD *)v12 + 2) = *(_QWORD *)(a3 + 16);
    *((_QWORD *)v12 + 3) = v12 + 40;
    *(__m128i *)v12 = v13;
    if ( v14 == a3 + 40 )
    {
      *(__m128i *)(v12 + 40) = _mm_loadu_si128((const __m128i *)(a3 + 40));
    }
    else
    {
      *((_QWORD *)v12 + 3) = v14;
      *((_QWORD *)v12 + 5) = *(_QWORD *)(a3 + 40);
    }
    v15 = *(_QWORD *)(a3 + 32);
    v16 = _mm_loadu_si128((const __m128i *)(a3 + 56));
    *(_QWORD *)(a3 + 24) = a3 + 40;
    *(_QWORD *)(a3 + 32) = 0;
    v17 = *(_WORD *)(a3 + 72);
    *(_BYTE *)(a3 + 40) = 0;
    v18 = *(_BYTE *)(a3 + 74);
    *((_QWORD *)v12 + 4) = v15;
    *((_WORD *)v12 + 36) = v17;
    v12[74] = v18;
    *(__m128i *)(v12 + 56) = v16;
  }
  if ( a2 != (const __m128i *)v5 )
  {
    v19 = v37;
    for ( i = (const __m128i *)(v5 + 40); ; i += 5 )
    {
      if ( v19 )
      {
        *(__m128i *)v19 = _mm_loadu_si128((const __m128i *)((char *)i - 40));
        *(_QWORD *)(v19 + 16) = i[-2].m128i_i64[1];
        *(_QWORD *)(v19 + 24) = v19 + 40;
        v23 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v23 == i )
        {
          *(__m128i *)(v19 + 40) = _mm_loadu_si128(i);
        }
        else
        {
          *(_QWORD *)(v19 + 24) = v23;
          *(_QWORD *)(v19 + 40) = i->m128i_i64[0];
        }
        *(_QWORD *)(v19 + 32) = i[-1].m128i_i64[1];
        v21 = _mm_loadu_si128(i + 1);
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        *(__m128i *)(v19 + 56) = v21;
        *(_WORD *)(v19 + 72) = i[2].m128i_i16[0];
        *(_BYTE *)(v19 + 74) = i[2].m128i_i8[2];
      }
      v22 = i[-1].m128i_u64[0];
      if ( (const __m128i *)v22 != i )
        j_j___libc_free_0(v22);
      if ( a2 == (const __m128i *)&i[2].m128i_u64[1] )
        break;
      v19 += 80LL;
    }
    v11 = v19 + 160;
  }
  if ( a2 != (const __m128i *)v4 )
  {
    v24 = a2;
    v25 = v11;
    do
    {
      v28 = _mm_loadu_si128(v24);
      *(_QWORD *)(v25 + 16) = v24[1].m128i_i64[0];
      *(_QWORD *)(v25 + 24) = v25 + 40;
      v29 = (const __m128i *)v24[1].m128i_i64[1];
      *(__m128i *)v25 = v28;
      if ( v29 == (const __m128i *)&v24[2].m128i_u64[1] )
      {
        *(__m128i *)(v25 + 40) = _mm_loadu_si128((const __m128i *)((char *)v24 + 40));
      }
      else
      {
        *(_QWORD *)(v25 + 24) = v29;
        *(_QWORD *)(v25 + 40) = v24[2].m128i_i64[1];
      }
      v26 = v24[2].m128i_i64[0];
      v27 = _mm_loadu_si128((const __m128i *)((char *)v24 + 56));
      v24 += 5;
      v25 += 80;
      *(_QWORD *)(v25 - 48) = v26;
      LOWORD(v26) = v24[-1].m128i_i16[4];
      *(__m128i *)(v25 - 24) = v27;
      *(_WORD *)(v25 - 8) = v26;
      *(_BYTE *)(v25 - 6) = v24[-1].m128i_i8[10];
    }
    while ( v24 != (const __m128i *)v4 );
    v11 += 16
         * (5 * ((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)((char *)v24 - (char *)a2 - 80) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
          + 5);
  }
  if ( v5 )
  {
    v33 = v11;
    j_j___libc_free_0(v5);
    v11 = v33;
  }
  *a1 = v37;
  a1[1] = v11;
  a1[2] = v35;
  return a1;
}
