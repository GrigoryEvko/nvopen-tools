// Function: sub_1DD2ED0
// Address: 0x1dd2ed0
//
__m128i *__fastcall sub_1DD2ED0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __m128i *result; // rax
  __int64 v6; // r9
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // rcx
  __m128i *v11; // rax
  __int64 v12; // rdx
  __int32 v13; // esi
  __int64 v14; // rsi
  __int32 v15; // edi
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rsi
  int v22; // ecx
  int v23; // eax
  __int64 v24; // rcx
  __m128i *v25; // rbx
  __m128i *i; // rax
  __int32 v27; // edi
  __int32 v28; // ecx
  __int32 v29; // edx
  __int64 v30; // rdi
  __m128i *v31; // r15
  __int32 v32; // edi
  __int64 v33; // rsi
  int v34; // edi
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int32 v38; // esi
  __int64 v39; // rbx
  unsigned __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rax
  __m128i v46; // xmm1
  int v47; // eax
  __m128i v48; // [rsp-58h] [rbp-58h]
  __m128i v49; // [rsp-58h] [rbp-58h]

  result = (__m128i *)((char *)a2 - a1);
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  v6 = (__int64)a2;
  v8 = a1 + 24;
  v9 = a3;
  if ( !a3 )
  {
    v31 = a2;
    goto LABEL_50;
  }
  while ( 2 )
  {
    v10 = *(_QWORD *)(a1 + 32);
    --v9;
    v11 = (__m128i *)(a1
                    + 8
                    * (((__int64)(0xAAAAAAAAAAAAAAABLL * ((__int64)result >> 3)) >> 1)
                     + ((0xAAAAAAAAAAAAAAABLL * ((__int64)result >> 3)) & 0xFFFFFFFFFFFFFFFELL)));
    v12 = v11->m128i_i64[1];
    if ( v10 >= v12 )
    {
      if ( v10 != v12
        || (v13 = v11[1].m128i_i32[0], *(_DWORD *)(a1 + 40) >= v13)
        && (*(_DWORD *)(a1 + 40) != v13 || *(_DWORD *)(a1 + 44) >= v11[1].m128i_i32[1]) )
      {
        v33 = *(_QWORD *)(v6 - 16);
        if ( v10 < v33
          || v10 == v33
          && ((v34 = *(_DWORD *)(v6 - 8), *(_DWORD *)(a1 + 40) < v34)
           || *(_DWORD *)(a1 + 40) == v34 && *(_DWORD *)(a1 + 44) < *(_DWORD *)(v6 - 4)) )
        {
          v35 = *(_QWORD *)(a1 + 16);
          v36 = *(_QWORD *)a1;
          v19 = *(_QWORD *)(a1 + 8);
          *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 24));
LABEL_42:
          v37 = *(_QWORD *)(a1 + 40);
          *(_QWORD *)(a1 + 24) = v36;
          *(_QWORD *)(a1 + 32) = v19;
          *(_QWORD *)(a1 + 16) = v37;
          *(_QWORD *)(a1 + 40) = v35;
          v20 = *(_QWORD *)(v6 - 16);
          goto LABEL_15;
        }
        if ( v12 >= v33 )
        {
          if ( v12 != v33
            || (v38 = *(_DWORD *)(v6 - 8), v11[1].m128i_i32[0] >= v38)
            && (v11[1].m128i_i32[0] != v38 || v11[1].m128i_i32[1] >= *(_DWORD *)(v6 - 4)) )
          {
            v17 = *(_QWORD *)a1;
            v18 = *(_QWORD *)(a1 + 8);
            v16 = *(_QWORD *)(a1 + 16);
            *(__m128i *)a1 = _mm_loadu_si128(v11);
            goto LABEL_12;
          }
        }
LABEL_14:
        v21 = *(_QWORD *)a1;
        v20 = *(_QWORD *)(a1 + 8);
        v22 = *(_DWORD *)(a1 + 16);
        v23 = *(_DWORD *)(a1 + 20);
        *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(v6 - 24));
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(v6 - 8);
        *(_QWORD *)(v6 - 24) = v21;
        *(_QWORD *)(v6 - 16) = v20;
        *(_DWORD *)(v6 - 8) = v22;
        *(_DWORD *)(v6 - 4) = v23;
        v19 = *(_QWORD *)(a1 + 32);
        goto LABEL_15;
      }
    }
    v14 = *(_QWORD *)(v6 - 16);
    if ( v12 >= v14 )
    {
      if ( v12 != v14
        || (v15 = *(_DWORD *)(v6 - 8), v11[1].m128i_i32[0] >= v15)
        && (v11[1].m128i_i32[0] != v15 || v11[1].m128i_i32[1] >= *(_DWORD *)(v6 - 4)) )
      {
        if ( v10 >= v14 )
        {
          if ( v10 != v14
            || (v47 = *(_DWORD *)(v6 - 8), *(_DWORD *)(a1 + 40) >= v47)
            && (*(_DWORD *)(a1 + 40) != v47 || *(_DWORD *)(a1 + 44) >= *(_DWORD *)(v6 - 4)) )
          {
            v36 = *(_QWORD *)a1;
            v19 = *(_QWORD *)(a1 + 8);
            v35 = *(_QWORD *)(a1 + 16);
            *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 24));
            goto LABEL_42;
          }
        }
        goto LABEL_14;
      }
    }
    v16 = *(_QWORD *)(a1 + 16);
    v17 = *(_QWORD *)a1;
    v18 = *(_QWORD *)(a1 + 8);
    *(__m128i *)a1 = _mm_loadu_si128(v11);
LABEL_12:
    *(_QWORD *)(a1 + 16) = v11[1].m128i_i64[0];
    v11->m128i_i64[0] = v17;
    v11->m128i_i64[1] = v18;
    v11[1].m128i_i64[0] = v16;
    v19 = *(_QWORD *)(a1 + 32);
    v20 = *(_QWORD *)(v6 - 16);
LABEL_15:
    v24 = *(_QWORD *)(a1 + 8);
    v25 = (__m128i *)v8;
    i = (__m128i *)v6;
    while ( 1 )
    {
      v31 = v25;
      if ( v19 >= v24 )
      {
        if ( v19 != v24 )
          break;
        v32 = *(_DWORD *)(a1 + 16);
        if ( v25[1].m128i_i32[0] >= v32 && (v25[1].m128i_i32[0] != v32 || v25[1].m128i_i32[1] >= *(_DWORD *)(a1 + 20)) )
          break;
      }
LABEL_22:
      v19 = v25[2].m128i_i64[0];
      v25 = (__m128i *)((char *)v25 + 24);
    }
    for ( i = (__m128i *)((char *)i - 24); ; v20 = i->m128i_i64[1] )
    {
      if ( v24 >= v20 )
      {
        if ( v24 != v20 )
          break;
        v27 = i[1].m128i_i32[0];
        if ( *(_DWORD *)(a1 + 16) >= v27 && (*(_DWORD *)(a1 + 16) != v27 || *(_DWORD *)(a1 + 20) >= i[1].m128i_i32[1]) )
          break;
      }
      i = (__m128i *)((char *)i - 24);
    }
    if ( v25 < i )
    {
      v28 = v25[1].m128i_i32[0];
      v29 = v25[1].m128i_i32[1];
      v30 = v25->m128i_i64[0];
      *v25 = _mm_loadu_si128(i);
      v25[1].m128i_i64[0] = i[1].m128i_i64[0];
      i[1].m128i_i32[1] = v29;
      v20 = i[-1].m128i_i64[0];
      i->m128i_i64[0] = v30;
      i->m128i_i64[1] = v19;
      i[1].m128i_i32[0] = v28;
      v24 = *(_QWORD *)(a1 + 8);
      goto LABEL_22;
    }
    sub_1DD2ED0(v25, v6, v9);
    result = (__m128i *)((char *)v25 - a1);
    if ( (__int64)v25->m128i_i64 - a1 > 384 )
    {
      if ( v9 )
      {
        v6 = (__int64)v25;
        continue;
      }
LABEL_50:
      v39 = 0xAAAAAAAAAAAAAAABLL * ((__int64)result >> 3);
      v40 = (v39 - 2) & 0xFFFFFFFFFFFFFFFELL;
      v41 = (v39 - 2) >> 1;
      v48 = _mm_loadu_si128((const __m128i *)(a1 + 8 * (v41 + v40)));
      sub_1DD2CE0(a1, v41, v39, a4, a5, v6, v48.m128i_i64[0], v48.m128i_i64[1], *(_QWORD *)(a1 + 8 * (v41 + v40) + 16));
      do
      {
        --v41;
        v49 = _mm_loadu_si128((const __m128i *)(a1 + 24 * v41));
        sub_1DD2CE0(a1, v41, v39, v42, v43, v44, v49.m128i_i64[0], v49.m128i_i64[1], *(_QWORD *)(a1 + 24 * v41 + 16));
      }
      while ( v41 );
      do
      {
        v31 = (__m128i *)((char *)v31 - 24);
        v45 = v31[1].m128i_i64[0];
        v46 = _mm_loadu_si128(v31);
        *v31 = _mm_loadu_si128((const __m128i *)a1);
        v31[1].m128i_i64[0] = *(_QWORD *)(a1 + 16);
        result = sub_1DD2CE0(
                   a1,
                   0,
                   0xAAAAAAAAAAAAAAABLL * (((__int64)v31->m128i_i64 - a1) >> 3),
                   v42,
                   v43,
                   v44,
                   v46.m128i_i64[0],
                   v46.m128i_i64[1],
                   v45);
      }
      while ( (__int64)v31->m128i_i64 - a1 > 24 );
    }
    return result;
  }
}
