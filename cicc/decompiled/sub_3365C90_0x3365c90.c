// Function: sub_3365C90
// Address: 0x3365c90
//
__int64 __fastcall sub_3365C90(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __m128i *v7; // r10
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // r11
  __int32 v13; // r8d
  __int64 v14; // rdi
  __m128i *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rdx
  __m128i *v19; // r14
  _QWORD *v20; // rcx
  __m128i *v21; // rax
  __m128i *v22; // r13
  __int64 v23; // rdi
  __int32 v24; // edx
  __int64 v25; // rax
  __int64 v26; // rbx
  unsigned __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  __m128i v33; // xmm1
  __int64 v34; // [rsp-60h] [rbp-60h]
  __m128i v35; // [rsp-58h] [rbp-58h]
  __m128i v36; // [rsp-58h] [rbp-58h]

  result = (__int64)a2->m128i_i64 - a1;
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  v7 = a2;
  v9 = a3;
  if ( !a3 )
  {
    v22 = a2;
    goto LABEL_24;
  }
  v34 = a1 + 24;
  while ( 2 )
  {
    v10 = *(_QWORD *)(a1 + 24);
    v11 = *(_QWORD *)a1;
    --v9;
    v12 = *(_QWORD *)(a1 + 8);
    v13 = *(_DWORD *)(a1 + 16);
    v14 = v7[-2].m128i_i64[1];
    v15 = (__m128i *)(a1
                    + 8
                    * (((__int64)(0xAAAAAAAAAAAAAAABLL * (result >> 3)) >> 1)
                     + ((0xAAAAAAAAAAAAAAABLL * (result >> 3)) & 0xFFFFFFFFFFFFFFFELL)));
    v16 = v15->m128i_i64[0];
    if ( v10 >= v15->m128i_i64[0] )
    {
      if ( v10 >= v14 )
      {
        if ( v16 >= v14 )
          goto LABEL_6;
LABEL_20:
        v17 = *(_QWORD *)a1;
        *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v7 - 24));
        *(_QWORD *)(a1 + 16) = v7[-1].m128i_i64[1];
        v7[-2].m128i_i64[1] = v11;
        v7[-1].m128i_i64[0] = v12;
        v7[-1].m128i_i32[2] = v13;
        v11 = *(_QWORD *)(a1 + 24);
        goto LABEL_7;
      }
LABEL_18:
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 24));
      v25 = *(_QWORD *)(a1 + 40);
      *(_QWORD *)(a1 + 24) = v11;
      *(_QWORD *)(a1 + 32) = v12;
      *(_QWORD *)(a1 + 16) = v25;
      *(_DWORD *)(a1 + 40) = v13;
      v17 = v7[-2].m128i_i64[1];
      goto LABEL_7;
    }
    if ( v16 >= v14 )
    {
      if ( v10 < v14 )
        goto LABEL_20;
      goto LABEL_18;
    }
LABEL_6:
    *(__m128i *)a1 = _mm_loadu_si128(v15);
    *(_QWORD *)(a1 + 16) = v15[1].m128i_i64[0];
    v15->m128i_i64[0] = v11;
    v15->m128i_i64[1] = v12;
    v15[1].m128i_i32[0] = v13;
    v17 = v7[-2].m128i_i64[1];
    v11 = *(_QWORD *)(a1 + 24);
LABEL_7:
    v18 = *(_QWORD *)a1;
    v19 = (__m128i *)v34;
    v20 = (_QWORD *)(a1 + 48);
    v21 = v7;
    while ( 1 )
    {
      v22 = v19;
      if ( v11 < v18 )
        goto LABEL_13;
      v21 = (__m128i *)((char *)v21 - 24);
      if ( v17 > v18 )
      {
        do
          v21 = (__m128i *)((char *)v21 - 24);
        while ( v21->m128i_i64[0] > v18 );
      }
      if ( v21 <= v19 )
        break;
      v23 = *(v20 - 2);
      v24 = *((_DWORD *)v20 - 2);
      *(__m128i *)(v20 - 3) = _mm_loadu_si128(v21);
      *(v20 - 1) = v21[1].m128i_i64[0];
      v21->m128i_i64[1] = v23;
      v17 = v21[-2].m128i_i64[1];
      v21->m128i_i64[0] = v11;
      v21[1].m128i_i32[0] = v24;
      v18 = *(_QWORD *)a1;
LABEL_13:
      v11 = *v20;
      v19 = (__m128i *)((char *)v19 + 24);
      v20 += 3;
    }
    sub_3365C90(v19, v7, v9, v20);
    result = (__int64)v19->m128i_i64 - a1;
    if ( (__int64)v19->m128i_i64 - a1 > 384 )
    {
      if ( v9 )
      {
        v7 = v19;
        continue;
      }
LABEL_24:
      v26 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
      v27 = (v26 - 2) & 0xFFFFFFFFFFFFFFFELL;
      v28 = (v26 - 2) >> 1;
      v35 = _mm_loadu_si128((const __m128i *)(a1 + 8 * (v28 + v27)));
      sub_3365330(a1, v28, v26, a4, a5, a6, v35.m128i_i64[0], v35.m128i_i64[1], *(_QWORD *)(a1 + 8 * (v28 + v27) + 16));
      do
      {
        --v28;
        v36 = _mm_loadu_si128((const __m128i *)(a1 + 24 * v28));
        sub_3365330(a1, v28, v26, v29, v30, v31, v36.m128i_i64[0], v36.m128i_i64[1], *(_QWORD *)(a1 + 24 * v28 + 16));
      }
      while ( v28 );
      do
      {
        v22 = (__m128i *)((char *)v22 - 24);
        v32 = v22[1].m128i_i64[0];
        v33 = _mm_loadu_si128(v22);
        *v22 = _mm_loadu_si128((const __m128i *)a1);
        v22[1].m128i_i64[0] = *(_QWORD *)(a1 + 16);
        result = sub_3365330(
                   a1,
                   0,
                   0xAAAAAAAAAAAAAAABLL * (((__int64)v22->m128i_i64 - a1) >> 3),
                   v29,
                   v30,
                   v31,
                   v33.m128i_i64[0],
                   v33.m128i_i64[1],
                   v32);
      }
      while ( (__int64)v22->m128i_i64 - a1 > 24 );
    }
    return result;
  }
}
