// Function: sub_3218FB0
// Address: 0x3218fb0
//
__int64 __fastcall sub_3218FB0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // r9
  __m128i *v8; // r13
  __int64 v9; // r12
  __int64 v10; // r11
  __int64 v11; // r10
  unsigned int v12; // edi
  __m128i *v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // esi
  unsigned int v16; // ecx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  unsigned int v21; // ecx
  __m128i *v22; // rbx
  __m128i *v23; // rax
  __m128i *v24; // r14
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  __int128 v37; // xmm1

  result = (__int64)a2->m128i_i64 - a1;
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  v6 = (__int64)a2;
  v8 = (__m128i *)(a1 + 24);
  v9 = a3;
  if ( !a3 )
  {
    v24 = a2;
    goto LABEL_23;
  }
  while ( 2 )
  {
    v10 = *(_QWORD *)a1;
    v11 = *(_QWORD *)(a1 + 8);
    --v9;
    v12 = *(_DWORD *)(*(_QWORD *)(v6 - 8) + 16LL);
    v13 = (__m128i *)(a1
                    + 8
                    * (((__int64)(0xAAAAAAAAAAAAAAABLL * (result >> 3)) >> 1)
                     + ((0xAAAAAAAAAAAAAAABLL * (result >> 3)) & 0xFFFFFFFFFFFFFFFELL)));
    v14 = *(_QWORD *)(a1 + 40);
    v15 = *(_DWORD *)(v14 + 16);
    v16 = *(_DWORD *)(v13[1].m128i_i64[0] + 16);
    if ( v15 >= v16 )
    {
      if ( v15 >= v12 )
      {
        if ( v16 >= v12 )
          goto LABEL_5;
LABEL_19:
        *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(v6 - 24));
        v30 = *(_QWORD *)(v6 - 8);
        *(_QWORD *)(v6 - 24) = v10;
        *(_QWORD *)(v6 - 16) = v11;
        v19 = *(_QWORD *)(a1 + 16);
        *(_QWORD *)(a1 + 16) = v30;
        *(_QWORD *)(v6 - 8) = v19;
        v20 = *(_QWORD *)(a1 + 40);
        v14 = *(_QWORD *)(a1 + 16);
        goto LABEL_6;
      }
LABEL_17:
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 24));
      v20 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 24) = v10;
      *(_QWORD *)(a1 + 32) = v11;
      *(_QWORD *)(a1 + 16) = v14;
      *(_QWORD *)(a1 + 40) = v20;
      v19 = *(_QWORD *)(v6 - 8);
      goto LABEL_6;
    }
    if ( v16 >= v12 )
    {
      if ( v15 < v12 )
        goto LABEL_19;
      goto LABEL_17;
    }
LABEL_5:
    *(__m128i *)a1 = _mm_loadu_si128(v13);
    v13->m128i_i64[0] = v10;
    v17 = v13[1].m128i_i64[0];
    v13->m128i_i64[1] = v11;
    v18 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = v17;
    v13[1].m128i_i64[0] = v18;
    v19 = *(_QWORD *)(v6 - 8);
    v20 = *(_QWORD *)(a1 + 40);
    v14 = *(_QWORD *)(a1 + 16);
LABEL_6:
    v21 = *(_DWORD *)(v14 + 16);
    v22 = v8;
    v23 = (__m128i *)v6;
    while ( 1 )
    {
      v24 = v22;
      if ( *(_DWORD *)(v20 + 16) < v21 )
        goto LABEL_12;
      v23 = (__m128i *)((char *)v23 - 24);
      if ( v21 < *(_DWORD *)(v19 + 16) )
      {
        do
        {
          v25 = v23[-1].m128i_i64[1];
          v23 = (__m128i *)((char *)v23 - 24);
        }
        while ( *(_DWORD *)(v25 + 16) > v21 );
      }
      if ( v23 <= v22 )
        break;
      v26 = v22->m128i_i64[0];
      v27 = v22->m128i_i64[1];
      *v22 = _mm_loadu_si128(v23);
      v23->m128i_i64[0] = v26;
      v28 = v23[1].m128i_i64[0];
      v23->m128i_i64[1] = v27;
      v29 = v22[1].m128i_i64[0];
      v22[1].m128i_i64[0] = v28;
      v19 = v23[-1].m128i_i64[1];
      v23[1].m128i_i64[0] = v29;
      v21 = *(_DWORD *)(*(_QWORD *)(a1 + 16) + 16LL);
LABEL_12:
      v20 = v22[2].m128i_i64[1];
      v22 = (__m128i *)((char *)v22 + 24);
    }
    sub_3218FB0(v22, v6, v9);
    result = (__int64)v22->m128i_i64 - a1;
    if ( (__int64)v22->m128i_i64 - a1 > 384 )
    {
      if ( v9 )
      {
        v6 = (__int64)v22;
        continue;
      }
LABEL_23:
      v31 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
      v32 = (v31 - 2) >> 1;
      sub_3218310(
        a1,
        v32,
        v31,
        a4,
        a5,
        v6,
        *(_OWORD *)&_mm_loadu_si128((const __m128i *)(a1 + 8 * (v32 + ((v31 - 2) & 0xFFFFFFFFFFFFFFFELL)))),
        *(_QWORD *)(a1 + 8 * (v32 + ((v31 - 2) & 0xFFFFFFFFFFFFFFFELL)) + 16));
      do
      {
        --v32;
        sub_3218310(
          a1,
          v32,
          v31,
          v33,
          v34,
          v35,
          *(_OWORD *)&_mm_loadu_si128((const __m128i *)(a1 + 24 * v32)),
          *(_QWORD *)(a1 + 24 * v32 + 16));
      }
      while ( v32 );
      do
      {
        v24 = (__m128i *)((char *)v24 - 24);
        v36 = v24[1].m128i_i64[0];
        v37 = (__int128)_mm_loadu_si128(v24);
        *v24 = _mm_loadu_si128((const __m128i *)a1);
        v24[1].m128i_i64[0] = *(_QWORD *)(a1 + 16);
        result = sub_3218310(
                   a1,
                   0,
                   0xAAAAAAAAAAAAAAABLL * (((__int64)v24->m128i_i64 - a1) >> 3),
                   v33,
                   v34,
                   v35,
                   v37,
                   v36);
      }
      while ( (__int64)v24->m128i_i64 - a1 > 24 );
    }
    return result;
  }
}
