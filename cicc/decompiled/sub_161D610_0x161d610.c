// Function: sub_161D610
// Address: 0x161d610
//
__int64 __fastcall sub_161D610(char *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // r9
  __m128i *v8; // r13
  __int64 v9; // r12
  unsigned __int64 v10; // rsi
  __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  char *v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rsi
  __m128i *v20; // rbx
  __m128i *v21; // rax
  __m128i *v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rdx
  unsigned __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // r11
  unsigned __int64 v41; // rax
  __m128i v42; // xmm0
  __int64 v43; // r11
  __m128i v44; // [rsp-58h] [rbp-58h]
  __m128i v45; // [rsp-58h] [rbp-58h]

  result = (char *)a2 - a1;
  if ( (char *)a2 - a1 <= 384 )
    return result;
  v6 = (__int64)a2;
  v8 = (__m128i *)(a1 + 24);
  v9 = a3;
  if ( !a3 )
  {
    v22 = a2;
    goto LABEL_22;
  }
  while ( 2 )
  {
    v10 = *(_QWORD *)(v6 - 8);
    --v9;
    v11 = *(_QWORD *)a1;
    v12 = *((_QWORD *)a1 + 5);
    v13 = &a1[8 * ((__int64)(0xAAAAAAAAAAAAAAABLL * (result >> 3)) >> 1)
            + 8 * ((0xAAAAAAAAAAAAAAABLL * (result >> 3)) & 0xFFFFFFFFFFFFFFFELL)];
    v14 = *((_QWORD *)v13 + 2);
    if ( v12 >= v14 )
    {
      if ( v12 < v10 )
        goto LABEL_6;
      if ( v14 < v10 )
      {
LABEL_16:
        *(_QWORD *)a1 = *(_QWORD *)(v6 - 24);
        v28 = *(_QWORD *)(v6 - 16);
        *(_QWORD *)(v6 - 24) = v11;
        v29 = *((_QWORD *)a1 + 1);
        *((_QWORD *)a1 + 1) = v28;
        *(_QWORD *)(v6 - 16) = v29;
        v19 = *((_QWORD *)a1 + 2);
        *((_QWORD *)a1 + 2) = *(_QWORD *)(v6 - 8);
        *(_QWORD *)(v6 - 8) = v19;
        v18 = *((_QWORD *)a1 + 5);
        v12 = *((_QWORD *)a1 + 2);
        goto LABEL_7;
      }
LABEL_20:
      *(_QWORD *)a1 = *(_QWORD *)v13;
      v30 = *((_QWORD *)v13 + 1);
      *(_QWORD *)v13 = v11;
      v31 = *((_QWORD *)a1 + 1);
      *((_QWORD *)a1 + 1) = v30;
      v32 = *((_QWORD *)v13 + 2);
      *((_QWORD *)v13 + 1) = v31;
      v33 = *((_QWORD *)a1 + 2);
      *((_QWORD *)a1 + 2) = v32;
      *((_QWORD *)v13 + 2) = v33;
      v19 = *(_QWORD *)(v6 - 8);
      v18 = *((_QWORD *)a1 + 5);
      v12 = *((_QWORD *)a1 + 2);
      goto LABEL_7;
    }
    if ( v14 < v10 )
      goto LABEL_20;
    if ( v12 < v10 )
      goto LABEL_16;
LABEL_6:
    v15 = *((_QWORD *)a1 + 3);
    v16 = *((_QWORD *)a1 + 4);
    *((_QWORD *)a1 + 3) = v11;
    *(_QWORD *)a1 = v15;
    v17 = *((_QWORD *)a1 + 1);
    *((_QWORD *)a1 + 1) = v16;
    v18 = *((_QWORD *)a1 + 2);
    *((_QWORD *)a1 + 4) = v17;
    *((_QWORD *)a1 + 2) = v12;
    *((_QWORD *)a1 + 5) = v18;
    v19 = *(_QWORD *)(v6 - 8);
LABEL_7:
    v20 = v8;
    v21 = (__m128i *)v6;
    while ( 1 )
    {
      v22 = v20;
      if ( v18 < v12 )
        goto LABEL_13;
      v21 = (__m128i *)((char *)v21 - 24);
      if ( v19 > v12 )
      {
        do
          v21 = (__m128i *)((char *)v21 - 24);
        while ( v21[1].m128i_i64[0] > v12 );
      }
      if ( v20 >= v21 )
        break;
      v23 = v20->m128i_i64[0];
      v20->m128i_i64[0] = v21->m128i_i64[0];
      v24 = v21->m128i_i64[1];
      v21->m128i_i64[0] = v23;
      v25 = v20->m128i_i64[1];
      v20->m128i_i64[1] = v24;
      v26 = v21[1].m128i_i64[0];
      v21->m128i_i64[1] = v25;
      v27 = v20[1].m128i_i64[0];
      v20[1].m128i_i64[0] = v26;
      v19 = v21[-1].m128i_u64[1];
      v21[1].m128i_i64[0] = v27;
      v12 = *((_QWORD *)a1 + 2);
LABEL_13:
      v18 = v20[2].m128i_u64[1];
      v20 = (__m128i *)((char *)v20 + 24);
    }
    sub_161D610(v20, v6, v9);
    result = (char *)v20 - a1;
    if ( (char *)v20 - a1 > 384 )
    {
      if ( v9 )
      {
        v6 = (__int64)v20;
        continue;
      }
LABEL_22:
      v34 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
      v35 = (v34 - 2) & 0xFFFFFFFFFFFFFFFELL;
      v36 = (v34 - 2) >> 1;
      v44 = _mm_loadu_si128((const __m128i *)&a1[8 * v36 + 8 * v35]);
      sub_161CD90(
        (__int64)a1,
        v36,
        v34,
        a4,
        a5,
        v6,
        v44.m128i_i64[0],
        v44.m128i_i64[1],
        *(_QWORD *)&a1[8 * v36 + 16 + 8 * v35]);
      do
      {
        --v36;
        v45 = _mm_loadu_si128((const __m128i *)&a1[24 * v36]);
        sub_161CD90(
          (__int64)a1,
          v36,
          v40,
          v37,
          v38,
          v39,
          v45.m128i_i64[0],
          v45.m128i_i64[1],
          *(_QWORD *)&a1[24 * v36 + 16]);
      }
      while ( v36 );
      do
      {
        v41 = v22[-1].m128i_u64[1];
        v22 = (__m128i *)((char *)v22 - 24);
        v42 = _mm_loadu_si128(v22);
        v22->m128i_i64[0] = *(_QWORD *)a1;
        v22->m128i_i64[1] = *((_QWORD *)a1 + 1);
        v22[1].m128i_i64[0] = *((_QWORD *)a1 + 2);
        result = (__int64)sub_161CD90(
                            (__int64)a1,
                            0,
                            0xAAAAAAAAAAAAAAABLL * (((char *)v22 - a1) >> 3),
                            v37,
                            v38,
                            v39,
                            v42.m128i_i64[0],
                            v42.m128i_i64[1],
                            v41);
      }
      while ( v43 > 24 );
    }
    return result;
  }
}
