// Function: sub_1A1B940
// Address: 0x1a1b940
//
__m128i *__fastcall sub_1A1B940(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *result; // rax
  __m128i *v7; // r10
  __m128i *v9; // r12
  __int64 v10; // r14
  unsigned __int64 v11; // rcx
  __m128i *v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rcx
  unsigned __int64 v20; // r9
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rsi
  unsigned __int64 *v23; // rdi
  __m128i *v24; // r15
  __m128i *i; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rax
  unsigned __int64 v33; // rsi
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rbx
  unsigned __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rax
  __m128i v46; // xmm1
  unsigned __int64 *v47; // [rsp-60h] [rbp-60h]
  __m128i v48; // [rsp-58h] [rbp-58h]
  __m128i v49; // [rsp-58h] [rbp-58h]

  result = (__m128i *)((char *)a2 - a1);
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  v7 = a2;
  if ( !a3 )
  {
    v24 = a2;
    goto LABEL_54;
  }
  v9 = (__m128i *)(a1 + 24);
  v10 = a3;
  v47 = (unsigned __int64 *)(a1 + 48);
  while ( 2 )
  {
    v11 = *(_QWORD *)(a1 + 24);
    --v10;
    v12 = (__m128i *)(a1
                    + 8
                    * (((__int64)(0xAAAAAAAAAAAAAAABLL * ((__int64)result >> 3)) >> 1)
                     + ((0xAAAAAAAAAAAAAAABLL * ((__int64)result >> 3)) & 0xFFFFFFFFFFFFFFFELL)));
    v13 = v12->m128i_i64[0];
    if ( v11 < v12->m128i_i64[0] )
      goto LABEL_8;
    if ( v11 > v12->m128i_i64[0] )
      goto LABEL_42;
    v14 = (*(__int64 *)(a1 + 40) >> 2) & 1;
    if ( (_BYTE)v14 == ((v12[1].m128i_i64[0] >> 2) & 1) )
    {
      if ( *(_QWORD *)(a1 + 32) > v12->m128i_i64[1] )
        goto LABEL_8;
LABEL_42:
      v33 = v7[-2].m128i_u64[1];
      if ( v11 >= v33 )
      {
        if ( v11 <= v33 )
        {
          v34 = (*(__int64 *)(a1 + 40) >> 2) & 1;
          if ( (_BYTE)v34 == ((v7[-1].m128i_i64[1] >> 2) & 1) )
          {
            if ( *(_QWORD *)(a1 + 32) > v7[-1].m128i_i64[0] )
              goto LABEL_46;
          }
          else if ( !(_BYTE)v34 )
          {
            goto LABEL_46;
          }
        }
        if ( v13 >= v33 )
        {
          if ( v13 <= v33 )
          {
            v38 = (v12[1].m128i_i64[0] >> 2) & 1;
            if ( (_BYTE)v38 == ((v7[-1].m128i_i64[1] >> 2) & 1) )
            {
              if ( v12->m128i_i64[1] > (unsigned __int64)v7[-1].m128i_i64[0] )
                goto LABEL_33;
            }
            else if ( !(_BYTE)v38 )
            {
              goto LABEL_33;
            }
          }
          v18 = *(_QWORD *)a1;
          v19 = *(_QWORD *)(a1 + 8);
          v17 = *(_QWORD *)(a1 + 16);
          *(__m128i *)a1 = _mm_loadu_si128(v12);
          goto LABEL_13;
        }
LABEL_33:
        v21 = *(_QWORD *)a1;
        v31 = *(_QWORD *)(a1 + 8);
        v32 = *(_QWORD *)(a1 + 16);
        *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v7 - 24));
        *(_QWORD *)(a1 + 16) = v7[-1].m128i_i64[1];
        v7[-2].m128i_i64[1] = v21;
        v7[-1].m128i_i64[0] = v31;
        v7[-1].m128i_i64[1] = v32;
        v20 = *(_QWORD *)(a1 + 24);
        goto LABEL_14;
      }
LABEL_46:
      v35 = *(_QWORD *)(a1 + 16);
      v20 = *(_QWORD *)a1;
      v36 = *(_QWORD *)(a1 + 8);
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 24));
      goto LABEL_47;
    }
    if ( (_BYTE)v14 )
      goto LABEL_42;
LABEL_8:
    v15 = v7[-2].m128i_u64[1];
    if ( v13 < v15 )
    {
LABEL_12:
      v17 = *(_QWORD *)(a1 + 16);
      v18 = *(_QWORD *)a1;
      v19 = *(_QWORD *)(a1 + 8);
      *(__m128i *)a1 = _mm_loadu_si128(v12);
LABEL_13:
      *(_QWORD *)(a1 + 16) = v12[1].m128i_i64[0];
      v12->m128i_i64[0] = v18;
      v12->m128i_i64[1] = v19;
      v12[1].m128i_i64[0] = v17;
      v20 = *(_QWORD *)(a1 + 24);
      v21 = v7[-2].m128i_u64[1];
      goto LABEL_14;
    }
    if ( v13 <= v15 )
    {
      v16 = (v12[1].m128i_i64[0] >> 2) & 1;
      if ( (_BYTE)v16 == ((v7[-1].m128i_i64[1] >> 2) & 1) )
      {
        if ( v12->m128i_i64[1] > (unsigned __int64)v7[-1].m128i_i64[0] )
          goto LABEL_12;
      }
      else if ( !(_BYTE)v16 )
      {
        goto LABEL_12;
      }
    }
    if ( v11 < v15 )
      goto LABEL_33;
    if ( v11 <= v15 )
    {
      v30 = (*(__int64 *)(a1 + 40) >> 2) & 1;
      if ( (_BYTE)v30 == ((v7[-1].m128i_i64[1] >> 2) & 1) )
      {
        if ( *(_QWORD *)(a1 + 32) > v7[-1].m128i_i64[0] )
          goto LABEL_33;
      }
      else if ( !(_BYTE)v30 )
      {
        goto LABEL_33;
      }
    }
    v20 = *(_QWORD *)a1;
    v36 = *(_QWORD *)(a1 + 8);
    v35 = *(_QWORD *)(a1 + 16);
    *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 24));
LABEL_47:
    v37 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 24) = v20;
    *(_QWORD *)(a1 + 32) = v36;
    *(_QWORD *)(a1 + 16) = v37;
    *(_QWORD *)(a1 + 40) = v35;
    v21 = v7[-2].m128i_u64[1];
LABEL_14:
    v22 = *(_QWORD *)a1;
    v23 = v47;
    v24 = v9;
    i = v7;
    while ( 1 )
    {
      if ( v20 < v22 )
        goto LABEL_17;
      if ( v20 > v22 )
        goto LABEL_20;
      v26 = ((__int64)*(v23 - 1) >> 2) & 1;
      if ( (_BYTE)v26 == ((*(__int64 *)(a1 + 16) >> 2) & 1) )
        break;
      if ( (_BYTE)v26 )
        goto LABEL_20;
LABEL_17:
      v20 = *v23;
      v24 = (__m128i *)((char *)v24 + 24);
      v23 += 3;
    }
    if ( *(v23 - 2) > *(_QWORD *)(a1 + 8) )
      goto LABEL_17;
LABEL_20:
    for ( i = (__m128i *)((char *)i - 24); ; i = (__m128i *)((char *)i - 24) )
    {
      if ( v21 > v22 )
        goto LABEL_23;
      if ( v21 < v22 )
      {
LABEL_26:
        if ( v24 >= i )
          goto LABEL_36;
LABEL_27:
        v28 = *(v23 - 1);
        v29 = *(v23 - 2);
        *(__m128i *)(v23 - 3) = _mm_loadu_si128(i);
        *(v23 - 1) = i[1].m128i_u64[0];
        i[1].m128i_i64[0] = v28;
        v21 = i[-2].m128i_u64[1];
        i->m128i_i64[0] = v20;
        i->m128i_i64[1] = v29;
        v22 = *(_QWORD *)a1;
        goto LABEL_17;
      }
      v27 = (*(__int64 *)(a1 + 16) >> 2) & 1;
      if ( (_BYTE)v27 == ((i[1].m128i_i64[0] >> 2) & 1) )
        break;
      if ( (_BYTE)v27 )
        goto LABEL_26;
LABEL_23:
      v21 = i[-2].m128i_u64[1];
    }
    if ( *(_QWORD *)(a1 + 8) > i->m128i_i64[1] )
      goto LABEL_23;
    if ( v24 < i )
      goto LABEL_27;
LABEL_36:
    sub_1A1B940(v24, v7, v10);
    result = (__m128i *)((char *)v24 - a1);
    if ( (__int64)v24->m128i_i64 - a1 > 384 )
    {
      if ( v10 )
      {
        v7 = v24;
        continue;
      }
LABEL_54:
      v39 = 0xAAAAAAAAAAAAAAABLL * ((__int64)result >> 3);
      v40 = (v39 - 2) & 0xFFFFFFFFFFFFFFFELL;
      v41 = (v39 - 2) >> 1;
      v48 = _mm_loadu_si128((const __m128i *)(a1 + 8 * (v41 + v40)));
      sub_1A1B050(a1, v41, v39, a4, a5, a6, v48.m128i_u64[0], v48.m128i_u64[1], *(_QWORD *)(a1 + 8 * (v41 + v40) + 16));
      do
      {
        --v41;
        v49 = _mm_loadu_si128((const __m128i *)(a1 + 24 * v41));
        sub_1A1B050(a1, v41, v39, v42, v43, v44, v49.m128i_u64[0], v49.m128i_u64[1], *(_QWORD *)(a1 + 24 * v41 + 16));
      }
      while ( v41 );
      do
      {
        v24 = (__m128i *)((char *)v24 - 24);
        v45 = v24[1].m128i_i64[0];
        v46 = _mm_loadu_si128(v24);
        *v24 = _mm_loadu_si128((const __m128i *)a1);
        v24[1].m128i_i64[0] = *(_QWORD *)(a1 + 16);
        result = sub_1A1B050(
                   a1,
                   0,
                   0xAAAAAAAAAAAAAAABLL * (((__int64)v24->m128i_i64 - a1) >> 3),
                   v42,
                   v43,
                   v44,
                   v46.m128i_u64[0],
                   v46.m128i_u64[1],
                   v45);
      }
      while ( (__int64)v24->m128i_i64 - a1 > 24 );
    }
    return result;
  }
}
