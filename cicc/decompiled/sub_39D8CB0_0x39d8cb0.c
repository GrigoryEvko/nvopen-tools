// Function: sub_39D8CB0
// Address: 0x39d8cb0
//
unsigned __int64 *__fastcall sub_39D8CB0(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int8 *v11; // rdx
  __int64 v12; // r8
  bool v13; // zf
  __m128i *v14; // rdx
  __m128i *v15; // r12
  _BYTE *v16; // rsi
  __m128i v17; // xmm7
  __int8 v18; // dl
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
  const __m128i *v35; // [rsp+8h] [rbp-58h]
  const __m128i *v36; // [rsp+10h] [rbp-50h]
  unsigned __int64 v37; // [rsp+18h] [rbp-48h]
  unsigned __int64 v39; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v5 - *a1) >> 4);
  if ( v7 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v5 - v6) >> 4);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x3333333333333333LL * ((__int64)(v5 - v6) >> 4);
  v11 = &a2->m128i_i8[-v6];
  if ( v9 )
  {
    v31 = 0x7FFFFFFFFFFFFFD0LL;
  }
  else
  {
    if ( !v10 )
    {
      v37 = 0;
      v12 = 80;
      v39 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x199999999999999LL )
      v10 = 0x199999999999999LL;
    v31 = 80 * v10;
  }
  v35 = a3;
  v32 = sub_22077B0(v31);
  v11 = &a2->m128i_i8[-v6];
  a3 = v35;
  v39 = v32;
  v12 = v32 + 80;
  v37 = v32 + v31;
LABEL_7:
  v13 = &v11[v39] == 0;
  v14 = (__m128i *)&v11[v39];
  v15 = v14;
  if ( !v13 )
  {
    v16 = (_BYTE *)a3[1].m128i_i64[1];
    v33 = v12;
    v36 = a3;
    *v14 = _mm_loadu_si128(a3);
    v14[1].m128i_i64[0] = a3[1].m128i_i64[0];
    v14[1].m128i_i64[1] = (__int64)&v14[2].m128i_i64[1];
    sub_39CF630(&v14[1].m128i_i64[1], v16, (__int64)&v16[a3[2].m128i_i64[0]]);
    v12 = v33;
    v17 = _mm_loadu_si128((const __m128i *)((char *)v36 + 56));
    v15[4].m128i_i32[2] = v36[4].m128i_i32[2];
    v18 = v36[4].m128i_i8[12];
    *(__m128i *)((char *)v15 + 56) = v17;
    v15[4].m128i_i8[12] = v18;
  }
  if ( a2 != (const __m128i *)v6 )
  {
    v19 = v39;
    for ( i = (const __m128i *)(v6 + 40); ; i += 5 )
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
        *(_DWORD *)(v19 + 72) = i[2].m128i_i32[0];
        *(_BYTE *)(v19 + 76) = i[2].m128i_i8[4];
      }
      v22 = i[-1].m128i_u64[0];
      if ( (const __m128i *)v22 != i )
        j_j___libc_free_0(v22);
      if ( a2 == (const __m128i *)&i[2].m128i_u64[1] )
        break;
      v19 += 80LL;
    }
    v12 = v19 + 160;
  }
  if ( a2 != (const __m128i *)v5 )
  {
    v24 = a2;
    v25 = v12;
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
      LODWORD(v26) = v24[-1].m128i_i32[2];
      *(__m128i *)(v25 - 24) = v27;
      *(_DWORD *)(v25 - 8) = v26;
      *(_BYTE *)(v25 - 4) = v24[-1].m128i_i8[12];
    }
    while ( v24 != (const __m128i *)v5 );
    v12 += 16
         * (5 * ((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)((char *)v24 - (char *)a2 - 80) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
          + 5);
  }
  if ( v6 )
  {
    v34 = v12;
    j_j___libc_free_0(v6);
    v12 = v34;
  }
  *a1 = v39;
  a1[1] = v12;
  a1[2] = v37;
  return a1;
}
