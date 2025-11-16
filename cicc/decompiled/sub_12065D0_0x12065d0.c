// Function: sub_12065D0
// Address: 0x12065d0
//
__int64 __fastcall sub_12065D0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __m128i *v7; // r14
  __int64 v8; // rbx
  __m128i *v9; // r9
  __m128i v10; // xmm0
  unsigned int v11; // ecx
  __m128i *v12; // rsi
  unsigned int v13; // edx
  unsigned int v14; // eax
  bool v15; // cf
  __int64 v16; // rax
  __int64 v17; // rdx
  __m128i v18; // xmm7
  __int64 v19; // rdx
  __m128i *v20; // r13
  __m128i *v21; // rcx
  unsigned int v22; // esi
  __m128i *v23; // rdx
  __m128i *v24; // rdx
  __int64 v25; // rax
  __m128i v26; // xmm0
  __int64 v27; // rbx
  __int64 i; // r12
  __m128i *v29; // r14
  __int64 v30; // rax
  __int64 v31; // r12
  __int128 v32; // xmm2
  __int64 v33; // rdx
  __int64 v34; // rax
  __m128i v35; // xmm5
  __m128i v36; // [rsp-58h] [rbp-58h]
  __int64 v37; // [rsp-48h] [rbp-48h]

  result = (__int64)a2->m128i_i64 - a1;
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  v7 = a2;
  v8 = a3;
  if ( !a3 )
    goto LABEL_24;
  v9 = a2;
  while ( 2 )
  {
    v10 = _mm_loadu_si128((const __m128i *)a1);
    --v8;
    v11 = v9[-2].m128i_i32[2] & 6;
    v36 = v10;
    v12 = (__m128i *)(a1
                    + 8
                    * ((__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)v9->m128i_i64 - a1) >> 3)) / 2
                     + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v9->m128i_i64 - a1) >> 3)
                       + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v9->m128i_i64 - a1) >> 3)) >> 63))
                      & 0xFFFFFFFFFFFFFFFELL)));
    v13 = *(_DWORD *)(a1 + 24) & 6;
    v14 = v12->m128i_i32[0] & 6;
    if ( v13 >= v14 )
    {
      if ( v13 < v11 )
      {
        v34 = *(_QWORD *)(a1 + 16);
        v35 = _mm_loadu_si128((const __m128i *)(a1 + 24));
        *(__m128i *)(a1 + 24) = v10;
        v37 = v34;
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a1 + 40);
        *(_QWORD *)(a1 + 40) = v34;
        *(__m128i *)a1 = v35;
        goto LABEL_8;
      }
      v15 = v14 < v11;
      v16 = *(_QWORD *)(a1 + 16);
      v37 = v16;
      if ( v15 )
        goto LABEL_23;
LABEL_22:
      *(__m128i *)a1 = _mm_loadu_si128(v12);
      *(_QWORD *)(a1 + 16) = v12[1].m128i_i64[0];
      v12[1].m128i_i64[0] = v16;
      *v12 = v10;
      goto LABEL_8;
    }
    v15 = v14 < v11;
    v16 = *(_QWORD *)(a1 + 16);
    v37 = v16;
    if ( v15 )
      goto LABEL_22;
    if ( v13 < v11 )
    {
LABEL_23:
      *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v9 - 24));
      *(_QWORD *)(a1 + 16) = v9[-1].m128i_i64[1];
      v9[-1].m128i_i64[1] = v16;
      *(__m128i *)((char *)v9 - 24) = v10;
      goto LABEL_8;
    }
    v17 = *(_QWORD *)(a1 + 40);
    v18 = _mm_loadu_si128((const __m128i *)(a1 + 24));
    *(_QWORD *)(a1 + 40) = v16;
    *(__m128i *)(a1 + 24) = v10;
    *(_QWORD *)(a1 + 16) = v17;
    *(__m128i *)a1 = v18;
LABEL_8:
    v19 = *(_QWORD *)a1;
    v20 = (__m128i *)(a1 + 24);
    v21 = v9;
    while ( 1 )
    {
      v7 = v20;
      v22 = v19 & 6;
      if ( (v20->m128i_i32[0] & 6u) < v22 )
        goto LABEL_15;
      v23 = (__m128i *)((char *)v21 - 24);
      if ( v22 >= (v21[-2].m128i_i32[2] & 6u) )
      {
        v21 = (__m128i *)((char *)v21 - 24);
        if ( v20 >= v23 )
          break;
        goto LABEL_14;
      }
      v24 = v21 - 3;
      do
      {
        v21 = v24;
        v24 = (__m128i *)((char *)v24 - 24);
      }
      while ( v22 < (v21->m128i_i32[0] & 6u) );
      if ( v20 >= v21 )
        break;
LABEL_14:
      v25 = v20[1].m128i_i64[0];
      v26 = _mm_loadu_si128(v20);
      *v20 = _mm_loadu_si128(v21);
      v37 = v25;
      v20[1].m128i_i64[0] = v21[1].m128i_i64[0];
      v21[1].m128i_i64[0] = v25;
      *v21 = v26;
      v19 = *(_QWORD *)a1;
      v36 = v26;
LABEL_15:
      v20 = (__m128i *)((char *)v20 + 24);
    }
    sub_12065D0(v20, v9, v8, v21, a5, v9, v36.m128i_i64[0], v36.m128i_i64[1], v37);
    result = (__int64)v20->m128i_i64 - a1;
    if ( (__int64)v20->m128i_i64 - a1 > 384 )
    {
      if ( v8 )
      {
        v9 = v20;
        continue;
      }
LABEL_24:
      v27 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
      for ( i = (v27 - 2) >> 1; ; --i )
      {
        sub_1205570(
          a1,
          i,
          v27,
          a4,
          a5,
          a6,
          *(_OWORD *)&_mm_loadu_si128((const __m128i *)(a1 + 24 * i)),
          *(_QWORD *)(a1 + 24 * i + 16));
        if ( !i )
          break;
      }
      v29 = (__m128i *)((char *)v7 - 24);
      do
      {
        v30 = v29[1].m128i_i64[0];
        v31 = (__int64)v29->m128i_i64 - a1;
        v32 = (__int128)_mm_loadu_si128(v29);
        *v29 = _mm_loadu_si128((const __m128i *)a1);
        v33 = (__int64)v29->m128i_i64 - a1;
        v29 = (__m128i *)((char *)v29 - 24);
        v29[2].m128i_i64[1] = *(_QWORD *)(a1 + 16);
        result = sub_1205570(a1, 0, 0xAAAAAAAAAAAAAAABLL * (v33 >> 3), a4, a5, a6, v32, v30);
      }
      while ( v31 > 24 );
    }
    return result;
  }
}
