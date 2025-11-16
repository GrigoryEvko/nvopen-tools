// Function: sub_2F14850
// Address: 0x2f14850
//
unsigned __int64 *__fastcall sub_2F14850(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  bool v6; // cf
  unsigned __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rax
  __m128i v11; // xmm6
  __int64 v12; // rcx
  __int64 v13; // rcx
  __m128i v14; // xmm7
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __m128i v18; // xmm6
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rbx
  const __m128i *v22; // r15
  unsigned __int64 v23; // r13
  const __m128i *v24; // rdx
  __m128i v25; // xmm2
  __int64 v26; // rdx
  __m128i v27; // xmm3
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 *v30; // r12
  unsigned __int64 *v31; // r14
  __int64 v32; // rcx
  const __m128i *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  __m128i v36; // xmm5
  __int64 v37; // rsi
  __int64 v38; // rsi
  __m128i v39; // xmm0
  __int64 v40; // rsi
  __m128i v41; // xmm4
  const __m128i *v42; // rsi
  unsigned __int64 v44; // rbx
  unsigned __int64 v45; // [rsp+0h] [rbp-60h]
  unsigned __int64 v47; // [rsp+10h] [rbp-50h]
  __int64 v48; // [rsp+18h] [rbp-48h]
  unsigned __int64 v49; // [rsp+20h] [rbp-40h]

  v49 = a1[1];
  v47 = *a1;
  v4 = 0x8E38E38E38E38E39LL * ((__int64)(v49 - *a1) >> 4);
  if ( v4 == 0xE38E38E38E38E3LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0x8E38E38E38E38E39LL * ((__int64)(v49 - *a1) >> 4);
  v6 = __CFADD__(v5, v4);
  v7 = v5 - 0x71C71C71C71C71C7LL * ((__int64)(v49 - *a1) >> 4);
  v8 = a2 - v47;
  if ( v6 )
  {
    v44 = 0x7FFFFFFFFFFFFFB0LL;
  }
  else
  {
    if ( !v7 )
    {
      v45 = 0;
      v9 = 144;
      v48 = 0;
      goto LABEL_7;
    }
    if ( v7 > 0xE38E38E38E38E3LL )
      v7 = 0xE38E38E38E38E3LL;
    v44 = 144 * v7;
  }
  v48 = sub_22077B0(v44);
  v45 = v48 + v44;
  v9 = v48 + 144;
LABEL_7:
  v10 = v48 + v8;
  if ( v48 + v8 )
  {
    v11 = _mm_loadu_si128((const __m128i *)a3);
    v12 = *(_QWORD *)(a3 + 24);
    *(_QWORD *)(v10 + 16) = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v10 + 24) = v10 + 40;
    *(__m128i *)v10 = v11;
    if ( v12 == a3 + 40 )
    {
      *(__m128i *)(v10 + 40) = _mm_loadu_si128((const __m128i *)(a3 + 40));
    }
    else
    {
      *(_QWORD *)(v10 + 24) = v12;
      *(_QWORD *)(v10 + 40) = *(_QWORD *)(a3 + 40);
    }
    v13 = *(_QWORD *)(a3 + 32);
    *(_QWORD *)(a3 + 24) = a3 + 40;
    v14 = _mm_loadu_si128((const __m128i *)(a3 + 56));
    *(_QWORD *)(v10 + 72) = v10 + 88;
    *(_QWORD *)(v10 + 32) = v13;
    v15 = *(_QWORD *)(a3 + 72);
    *(_QWORD *)(a3 + 32) = 0;
    *(_BYTE *)(a3 + 40) = 0;
    *(__m128i *)(v10 + 56) = v14;
    if ( v15 == a3 + 88 )
    {
      *(__m128i *)(v10 + 88) = _mm_loadu_si128((const __m128i *)(a3 + 88));
    }
    else
    {
      *(_QWORD *)(v10 + 72) = v15;
      *(_QWORD *)(v10 + 88) = *(_QWORD *)(a3 + 88);
    }
    *(_QWORD *)(a3 + 72) = a3 + 88;
    v16 = *(_QWORD *)(a3 + 120);
    v17 = *(_QWORD *)(a3 + 80);
    v18 = _mm_loadu_si128((const __m128i *)(a3 + 104));
    *(_QWORD *)(a3 + 80) = 0;
    *(_QWORD *)(v10 + 120) = v16;
    v19 = *(_QWORD *)(a3 + 128);
    *(_QWORD *)(v10 + 80) = v17;
    *(_QWORD *)(v10 + 128) = v19;
    v20 = *(_QWORD *)(a3 + 136);
    *(_BYTE *)(a3 + 88) = 0;
    *(_QWORD *)(v10 + 136) = v20;
    *(_QWORD *)(a3 + 136) = 0;
    *(_QWORD *)(a3 + 128) = 0;
    *(_QWORD *)(a3 + 120) = 0;
    *(__m128i *)(v10 + 104) = v18;
  }
  if ( a2 != v47 )
  {
    v21 = v48;
    v22 = (const __m128i *)(v47 + 40);
    v23 = v47 + 88;
    while ( 1 )
    {
      if ( v21 )
      {
        *(__m128i *)v21 = _mm_loadu_si128((const __m128i *)((char *)v22 - 40));
        *(_QWORD *)(v21 + 16) = v22[-2].m128i_i64[1];
        *(_QWORD *)(v21 + 24) = v21 + 40;
        v24 = (const __m128i *)v22[-1].m128i_i64[0];
        if ( v24 == v22 )
        {
          *(__m128i *)(v21 + 40) = _mm_loadu_si128(v22);
        }
        else
        {
          *(_QWORD *)(v21 + 24) = v24;
          *(_QWORD *)(v21 + 40) = v22->m128i_i64[0];
        }
        *(_QWORD *)(v21 + 32) = v22[-1].m128i_i64[1];
        v25 = _mm_loadu_si128(v22 + 1);
        v22[-1].m128i_i64[0] = (__int64)v22;
        v22[-1].m128i_i64[1] = 0;
        v22->m128i_i8[0] = 0;
        *(_QWORD *)(v21 + 72) = v21 + 88;
        *(__m128i *)(v21 + 56) = v25;
        v26 = v22[2].m128i_i64[0];
        if ( v26 == v23 )
        {
          *(__m128i *)(v21 + 88) = _mm_loadu_si128(v22 + 3);
        }
        else
        {
          *(_QWORD *)(v21 + 72) = v26;
          *(_QWORD *)(v21 + 88) = v22[3].m128i_i64[0];
        }
        *(_QWORD *)(v21 + 80) = v22[2].m128i_i64[1];
        v27 = _mm_loadu_si128(v22 + 4);
        v22[2].m128i_i64[0] = v23;
        v22[2].m128i_i64[1] = 0;
        v22[3].m128i_i8[0] = 0;
        *(__m128i *)(v21 + 104) = v27;
        *(_QWORD *)(v21 + 120) = v22[5].m128i_i64[0];
        *(_QWORD *)(v21 + 128) = v22[5].m128i_i64[1];
        *(_QWORD *)(v21 + 136) = v22[6].m128i_i64[0];
        v22[6].m128i_i64[0] = 0;
        v22[5].m128i_i64[1] = 0;
        v22[5].m128i_i64[0] = 0;
      }
      else
      {
        v30 = (unsigned __int64 *)v22[5].m128i_i64[1];
        v31 = (unsigned __int64 *)v22[5].m128i_i64[0];
        if ( v30 != v31 )
        {
          do
          {
            if ( (unsigned __int64 *)*v31 != v31 + 2 )
              j_j___libc_free_0(*v31);
            v31 += 6;
          }
          while ( v30 != v31 );
          v31 = (unsigned __int64 *)v22[5].m128i_i64[0];
        }
        if ( v31 )
          j_j___libc_free_0((unsigned __int64)v31);
      }
      v28 = v22[2].m128i_u64[0];
      if ( v28 != v23 )
        j_j___libc_free_0(v28);
      v29 = v22[-1].m128i_u64[0];
      if ( (const __m128i *)v29 != v22 )
        j_j___libc_free_0(v29);
      v23 += 144LL;
      if ( (unsigned __int64 *)a2 == &v22[6].m128i_u64[1] )
        break;
      v22 += 9;
      v21 += 144;
    }
    v9 = v21 + 288;
  }
  v32 = a2;
  if ( a2 != v49 )
  {
    v33 = (const __m128i *)(a2 + 40);
    v34 = v9;
    do
    {
      v41 = _mm_loadu_si128((const __m128i *)((char *)v33 - 40));
      *(_QWORD *)(v34 + 16) = v33[-2].m128i_i64[1];
      *(_QWORD *)(v34 + 24) = v34 + 40;
      v42 = (const __m128i *)v33[-1].m128i_i64[0];
      *(__m128i *)v34 = v41;
      if ( v42 == v33 )
      {
        *(__m128i *)(v34 + 40) = _mm_loadu_si128(v33);
      }
      else
      {
        *(_QWORD *)(v34 + 24) = v42;
        *(_QWORD *)(v34 + 40) = v33->m128i_i64[0];
      }
      v35 = v33[-1].m128i_i64[1];
      v36 = _mm_loadu_si128(v33 + 1);
      v33->m128i_i8[0] = 0;
      v33[-1].m128i_i64[0] = (__int64)v33;
      *(_QWORD *)(v34 + 32) = v35;
      *(_QWORD *)(v34 + 72) = v34 + 88;
      v37 = v33[2].m128i_i64[0];
      v33[-1].m128i_i64[1] = 0;
      *(__m128i *)(v34 + 56) = v36;
      if ( v37 == v32 + 88 )
      {
        *(__m128i *)(v34 + 88) = _mm_loadu_si128(v33 + 3);
      }
      else
      {
        *(_QWORD *)(v34 + 72) = v37;
        *(_QWORD *)(v34 + 88) = v33[3].m128i_i64[0];
      }
      v38 = v33[2].m128i_i64[1];
      v39 = _mm_loadu_si128(v33 + 4);
      v32 += 144;
      v34 += 144;
      v33 += 9;
      *(_QWORD *)(v34 - 64) = v38;
      v40 = v33[-4].m128i_i64[0];
      *(__m128i *)(v34 - 40) = v39;
      *(_QWORD *)(v34 - 24) = v40;
      *(_QWORD *)(v34 - 16) = v33[-4].m128i_i64[1];
      *(_QWORD *)(v34 - 8) = v33[-3].m128i_i64[0];
    }
    while ( v32 != v49 );
    v9 += 16 * (9 * ((0xE38E38E38E38E39LL * ((unsigned __int64)(v32 - a2 - 144) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 9);
  }
  if ( v47 )
    j_j___libc_free_0(v47);
  *a1 = v48;
  a1[1] = v9;
  a1[2] = v45;
  return a1;
}
