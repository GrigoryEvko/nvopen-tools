// Function: sub_39D5130
// Address: 0x39d5130
//
unsigned __int64 *__fastcall sub_39D5130(unsigned __int64 *a1, __int64 a2, const __m128i *a3)
{
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r8
  bool v12; // zf
  __m128i *v13; // rdx
  __m128i *v14; // r13
  _BYTE *v15; // rsi
  _BYTE *v16; // rsi
  __int64 v17; // rdx
  unsigned __int64 v18; // r15
  const __m128i *v19; // r14
  unsigned __int64 v20; // r13
  __m128i v21; // xmm2
  __int64 v22; // rsi
  __m128i v23; // xmm3
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  const __m128i *v26; // rsi
  const __m128i *v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rsi
  __m128i v31; // xmm5
  __int64 v32; // rsi
  __int64 v33; // rsi
  __m128i v34; // xmm0
  __m128i v35; // xmm4
  const __m128i *v36; // rsi
  unsigned __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // [rsp+0h] [rbp-60h]
  const __m128i *v41; // [rsp+0h] [rbp-60h]
  const __m128i *v42; // [rsp+8h] [rbp-58h]
  __int64 v43; // [rsp+8h] [rbp-58h]
  unsigned __int64 v44; // [rsp+10h] [rbp-50h]
  unsigned __int64 v46; // [rsp+20h] [rbp-40h]
  unsigned __int64 v47; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v46 = *a1;
  v6 = 0xEEEEEEEEEEEEEEEFLL * ((__int64)(v5 - *a1) >> 3);
  if ( v6 == 0x111111111111111LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xEEEEEEEEEEEEEEEFLL * ((__int64)(v5 - v46) >> 3);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x1111111111111111LL * ((__int64)(v5 - v46) >> 3);
  v10 = a2 - v46;
  if ( v8 )
  {
    v38 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v44 = 0;
      v11 = 120;
      v47 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x111111111111111LL )
      v9 = 0x111111111111111LL;
    v38 = 120 * v9;
  }
  v41 = a3;
  v39 = sub_22077B0(v38);
  v10 = a2 - v46;
  v47 = v39;
  a3 = v41;
  v11 = v39 + 120;
  v44 = v39 + v38;
LABEL_7:
  v12 = v47 + v10 == 0;
  v13 = (__m128i *)(v47 + v10);
  v14 = v13;
  if ( !v12 )
  {
    v15 = (_BYTE *)a3[1].m128i_i64[1];
    v40 = v11;
    v42 = a3;
    *v13 = _mm_loadu_si128(a3);
    v13[1].m128i_i64[0] = a3[1].m128i_i64[0];
    v13[1].m128i_i64[1] = (__int64)&v13[2].m128i_i64[1];
    sub_39CF630(&v13[1].m128i_i64[1], v15, (__int64)&v15[a3[2].m128i_i64[0]]);
    v14[4].m128i_i64[1] = (__int64)&v14[5].m128i_i64[1];
    v16 = (_BYTE *)v42[4].m128i_i64[1];
    v17 = v42[5].m128i_i64[0];
    *(__m128i *)((char *)v14 + 56) = _mm_loadu_si128((const __m128i *)((char *)v42 + 56));
    sub_39CF630(&v14[4].m128i_i64[1], v16, (__int64)&v16[v17]);
    v11 = v40;
    *(__m128i *)((char *)v14 + 104) = _mm_loadu_si128((const __m128i *)((char *)v42 + 104));
  }
  if ( a2 != v46 )
  {
    v18 = v47;
    v19 = (const __m128i *)(v46 + 40);
    v20 = v46 + 88;
    while ( 1 )
    {
      if ( v18 )
      {
        *(__m128i *)v18 = _mm_loadu_si128((const __m128i *)((char *)v19 - 40));
        *(_QWORD *)(v18 + 16) = v19[-2].m128i_i64[1];
        *(_QWORD *)(v18 + 24) = v18 + 40;
        v26 = (const __m128i *)v19[-1].m128i_i64[0];
        if ( v26 == v19 )
        {
          *(__m128i *)(v18 + 40) = _mm_loadu_si128(v19);
        }
        else
        {
          *(_QWORD *)(v18 + 24) = v26;
          *(_QWORD *)(v18 + 40) = v19->m128i_i64[0];
        }
        *(_QWORD *)(v18 + 32) = v19[-1].m128i_i64[1];
        v21 = _mm_loadu_si128(v19 + 1);
        v19[-1].m128i_i64[0] = (__int64)v19;
        v19[-1].m128i_i64[1] = 0;
        v19->m128i_i8[0] = 0;
        *(_QWORD *)(v18 + 72) = v18 + 88;
        *(__m128i *)(v18 + 56) = v21;
        v22 = v19[2].m128i_i64[0];
        if ( v22 == v20 )
        {
          *(__m128i *)(v18 + 88) = _mm_loadu_si128(v19 + 3);
        }
        else
        {
          *(_QWORD *)(v18 + 72) = v22;
          *(_QWORD *)(v18 + 88) = v19[3].m128i_i64[0];
        }
        *(_QWORD *)(v18 + 80) = v19[2].m128i_i64[1];
        v23 = _mm_loadu_si128(v19 + 4);
        v19[2].m128i_i64[0] = v20;
        v19[2].m128i_i64[1] = 0;
        v19[3].m128i_i8[0] = 0;
        *(__m128i *)(v18 + 104) = v23;
      }
      v24 = v19[2].m128i_u64[0];
      if ( v24 != v20 )
        j_j___libc_free_0(v24);
      v25 = v19[-1].m128i_u64[0];
      if ( (const __m128i *)v25 != v19 )
        j_j___libc_free_0(v25);
      v20 += 120LL;
      if ( (const __m128i *)a2 == &v19[5] )
        break;
      v19 = (const __m128i *)((char *)v19 + 120);
      v18 += 120LL;
    }
    v11 = v18 + 240;
  }
  if ( a2 != v5 )
  {
    v27 = (const __m128i *)(a2 + 40);
    v28 = a2;
    v29 = v11;
    do
    {
      v35 = _mm_loadu_si128((const __m128i *)((char *)v27 - 40));
      *(_QWORD *)(v29 + 16) = v27[-2].m128i_i64[1];
      *(_QWORD *)(v29 + 24) = v29 + 40;
      v36 = (const __m128i *)v27[-1].m128i_i64[0];
      *(__m128i *)v29 = v35;
      if ( v36 == v27 )
      {
        *(__m128i *)(v29 + 40) = _mm_loadu_si128(v27);
      }
      else
      {
        *(_QWORD *)(v29 + 24) = v36;
        *(_QWORD *)(v29 + 40) = v27->m128i_i64[0];
      }
      v30 = v27[-1].m128i_i64[1];
      v31 = _mm_loadu_si128(v27 + 1);
      v27->m128i_i8[0] = 0;
      v27[-1].m128i_i64[0] = (__int64)v27;
      *(_QWORD *)(v29 + 32) = v30;
      *(_QWORD *)(v29 + 72) = v29 + 88;
      v32 = v27[2].m128i_i64[0];
      v27[-1].m128i_i64[1] = 0;
      *(__m128i *)(v29 + 56) = v31;
      if ( v32 == v28 + 88 )
      {
        *(__m128i *)(v29 + 88) = _mm_loadu_si128(v27 + 3);
      }
      else
      {
        *(_QWORD *)(v29 + 72) = v32;
        *(_QWORD *)(v29 + 88) = v27[3].m128i_i64[0];
      }
      v33 = v27[2].m128i_i64[1];
      v28 += 120;
      v29 += 120;
      v27 = (const __m128i *)((char *)v27 + 120);
      v34 = _mm_loadu_si128((const __m128i *)((char *)v27 - 56));
      *(_QWORD *)(v29 - 40) = v33;
      *(__m128i *)(v29 - 16) = v34;
    }
    while ( v28 != v5 );
    v11 += 120 * (((0xEEEEEEEEEEEEEEFLL * ((unsigned __int64)(v28 - a2 - 120) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
  }
  if ( v46 )
  {
    v43 = v11;
    j_j___libc_free_0(v46);
    v11 = v43;
  }
  *a1 = v47;
  a1[1] = v11;
  a1[2] = v44;
  return a1;
}
