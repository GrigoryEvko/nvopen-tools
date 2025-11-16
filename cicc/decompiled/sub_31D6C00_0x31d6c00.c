// Function: sub_31D6C00
// Address: 0x31d6c00
//
__int64 __fastcall sub_31D6C00(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned __int32 v7; // r13d
  __m128i *v8; // rbx
  unsigned __int32 v9; // r12d
  unsigned __int32 v10; // r14d
  const void *v11; // rdx
  __int64 v12; // rax
  unsigned __int32 v13; // eax
  unsigned __int32 v14; // r12d
  __int64 v15; // r14
  unsigned __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  size_t v20; // r12
  size_t v21; // r14
  size_t v22; // rdx
  int v23; // eax
  size_t v24; // rcx
  size_t v25; // r9
  size_t v26; // rdx
  int v27; // eax
  size_t v28; // r14
  unsigned __int64 v29; // rcx
  size_t v30; // rdx
  int v31; // eax
  unsigned __int32 v32; // r14d
  __m128i v33; // xmm7
  const void *v34; // rdx
  __int64 v35; // rbx
  unsigned __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __m128i *v41; // r12
  __int64 v42; // rax
  __m128i v43; // xmm1
  size_t v44; // rcx
  size_t v45; // r10
  size_t v46; // rdx
  int v47; // eax
  __m128i v48; // xmm5
  const void *v49; // rdx
  size_t v50; // r12
  size_t v51; // rbx
  size_t v52; // rdx
  int v53; // eax
  __int64 v54; // rax
  const void *v55; // rdx
  size_t v56; // r12
  size_t v57; // r13
  size_t v58; // rdx
  int v59; // eax
  size_t v60; // r12
  size_t v61; // rcx
  size_t v62; // rdx
  int v63; // eax
  __int64 v64; // [rsp+10h] [rbp-80h]
  __int64 v65; // [rsp+18h] [rbp-78h]
  size_t v66; // [rsp+20h] [rbp-70h]
  __m128i *v67; // [rsp+28h] [rbp-68h]
  size_t v68; // [rsp+28h] [rbp-68h]
  unsigned __int64 v69; // [rsp+30h] [rbp-60h]
  size_t v70; // [rsp+30h] [rbp-60h]
  __int64 v71; // [rsp+30h] [rbp-60h]
  size_t v72; // [rsp+30h] [rbp-60h]
  size_t v73; // [rsp+30h] [rbp-60h]
  __m128i v74; // [rsp+40h] [rbp-50h]
  __m128i v75; // [rsp+40h] [rbp-50h]

  result = (__int64)a2->m128i_i64 - a1;
  v65 = (__int64)a2;
  v64 = a3;
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  if ( !a3 )
  {
    v67 = a2;
    goto LABEL_50;
  }
  while ( 2 )
  {
    v7 = *(_DWORD *)(a1 + 40);
    --v64;
    v8 = (__m128i *)(a1
                   + 8
                   * (((__int64)(0xAAAAAAAAAAAAAAABLL * (result >> 3)) >> 1)
                    + ((0xAAAAAAAAAAAAAAABLL * (result >> 3)) & 0xFFFFFFFFFFFFFFFELL)));
    v9 = v8[1].m128i_u32[0];
    if ( v7 <= v9 )
    {
      if ( v7 != v9 )
        goto LABEL_5;
      v28 = *(_QWORD *)(a1 + 32);
      v29 = v8->m128i_u64[1];
      v30 = v28;
      if ( v29 <= v28 )
        v30 = v8->m128i_u64[1];
      if ( !v30
        || (v71 = v8->m128i_i64[1],
            v31 = memcmp(*(const void **)(a1 + 24), (const void *)v8->m128i_i64[0], v30),
            v29 = v71,
            !v31) )
      {
        if ( v29 > v28 )
          goto LABEL_44;
LABEL_5:
        v10 = *(_DWORD *)(v65 - 8);
        if ( v7 <= v10 )
        {
          if ( v7 != v10 )
            goto LABEL_7;
          v44 = *(_QWORD *)(a1 + 32);
          v45 = *(_QWORD *)(v65 - 16);
          v46 = v44;
          if ( v45 <= v44 )
            v46 = *(_QWORD *)(v65 - 16);
          if ( v46
            && (v68 = *(_QWORD *)(a1 + 32),
                v72 = *(_QWORD *)(v65 - 16),
                v47 = memcmp(*(const void **)(a1 + 24), *(const void **)(v65 - 24), v46),
                v45 = v72,
                v44 = v68,
                v47) )
          {
            if ( v47 < 0 )
              goto LABEL_60;
          }
          else if ( v45 > v44 )
          {
            goto LABEL_60;
          }
LABEL_7:
          if ( v9 <= v10 )
          {
            if ( v9 != v10 )
            {
LABEL_9:
              v11 = *(const void **)a1;
              v12 = *(_QWORD *)(a1 + 8);
              *(__m128i *)a1 = _mm_loadu_si128(v8);
              v8->m128i_i64[0] = (__int64)v11;
              LODWORD(v11) = v8[1].m128i_i32[0];
              v8->m128i_i64[1] = v12;
              LODWORD(v12) = *(_DWORD *)(a1 + 16);
              *(_DWORD *)(a1 + 16) = (_DWORD)v11;
              v8[1].m128i_i32[0] = v12;
              v13 = *(_DWORD *)(a1 + 40);
              v7 = *(_DWORD *)(a1 + 16);
              v14 = *(_DWORD *)(v65 - 8);
              goto LABEL_10;
            }
            v56 = v8->m128i_u64[1];
            v57 = *(_QWORD *)(v65 - 16);
            v58 = v56;
            if ( v57 <= v56 )
              v58 = *(_QWORD *)(v65 - 16);
            if ( v58 && (v59 = memcmp((const void *)v8->m128i_i64[0], *(const void **)(v65 - 24), v58)) != 0 )
            {
              if ( v59 >= 0 )
                goto LABEL_9;
            }
            else if ( v57 <= v56 )
            {
              goto LABEL_9;
            }
          }
          goto LABEL_66;
        }
LABEL_60:
        v48 = _mm_loadu_si128((const __m128i *)(a1 + 24));
        v49 = *(const void **)a1;
        *(_QWORD *)(a1 + 32) = *(_QWORD *)(a1 + 8);
        v13 = *(_DWORD *)(a1 + 16);
        *(_QWORD *)(a1 + 24) = v49;
        *(_DWORD *)(a1 + 16) = v7;
        *(_DWORD *)(a1 + 40) = v13;
        *(__m128i *)a1 = v48;
        v14 = *(_DWORD *)(v65 - 8);
        goto LABEL_10;
      }
      if ( v31 >= 0 )
        goto LABEL_5;
    }
LABEL_44:
    v32 = *(_DWORD *)(v65 - 8);
    if ( v9 > v32 )
      goto LABEL_9;
    if ( v9 == v32 )
    {
      v60 = v8->m128i_u64[1];
      v61 = *(_QWORD *)(v65 - 16);
      v62 = v60;
      if ( v61 <= v60 )
        v62 = *(_QWORD *)(v65 - 16);
      if ( v62
        && (v73 = *(_QWORD *)(v65 - 16),
            v63 = memcmp((const void *)v8->m128i_i64[0], *(const void **)(v65 - 24), v62),
            v61 = v73,
            v63) )
      {
        if ( v63 < 0 )
          goto LABEL_9;
      }
      else if ( v61 > v60 )
      {
        goto LABEL_9;
      }
    }
    if ( v7 <= v32 )
    {
      if ( v7 != v32 )
      {
LABEL_48:
        v33 = _mm_loadu_si128((const __m128i *)(a1 + 24));
        v34 = *(const void **)a1;
        *(_QWORD *)(a1 + 32) = *(_QWORD *)(a1 + 8);
        v13 = *(_DWORD *)(a1 + 16);
        *(_QWORD *)(a1 + 24) = v34;
        *(_DWORD *)(a1 + 16) = v7;
        *(_DWORD *)(a1 + 40) = v13;
        *(__m128i *)a1 = v33;
        v14 = *(_DWORD *)(v65 - 8);
        goto LABEL_10;
      }
      v50 = *(_QWORD *)(a1 + 32);
      v51 = *(_QWORD *)(v65 - 16);
      v52 = v51;
      if ( v50 <= v51 )
        v52 = *(_QWORD *)(a1 + 32);
      if ( v52 && (v53 = memcmp(*(const void **)(a1 + 24), *(const void **)(v65 - 24), v52)) != 0 )
      {
        if ( v53 >= 0 )
          goto LABEL_48;
      }
      else if ( v50 >= v51 )
      {
        goto LABEL_48;
      }
    }
LABEL_66:
    v54 = *(_QWORD *)(a1 + 8);
    v55 = *(const void **)a1;
    *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(v65 - 24));
    *(_QWORD *)(v65 - 16) = v54;
    LODWORD(v54) = *(_DWORD *)(v65 - 8);
    *(_QWORD *)(v65 - 24) = v55;
    v14 = *(_DWORD *)(a1 + 16);
    *(_DWORD *)(a1 + 16) = v54;
    *(_DWORD *)(v65 - 8) = v14;
    v13 = *(_DWORD *)(a1 + 40);
    v7 = *(_DWORD *)(a1 + 16);
LABEL_10:
    v15 = a1 + 24;
    v16 = v65;
    while ( 1 )
    {
      v67 = (__m128i *)v15;
      if ( v13 > v7 )
        goto LABEL_14;
      if ( v13 != v7 )
        goto LABEL_17;
      v24 = *(_QWORD *)(v15 + 8);
      v25 = *(_QWORD *)(a1 + 8);
      v26 = v24;
      if ( v25 <= v24 )
        v26 = *(_QWORD *)(a1 + 8);
      if ( v26 )
      {
        v66 = *(_QWORD *)(v15 + 8);
        v70 = *(_QWORD *)(a1 + 8);
        v27 = memcmp(*(const void **)v15, *(const void **)a1, v26);
        v25 = v70;
        v24 = v66;
        if ( v27 )
          break;
      }
      if ( v25 <= v24 )
        goto LABEL_17;
LABEL_14:
      v13 = *(_DWORD *)(v15 + 40);
      v15 += 24;
    }
    if ( v27 < 0 )
      goto LABEL_14;
LABEL_17:
    v16 -= 24LL;
    v19 = v15;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( v14 < v7 )
          goto LABEL_19;
        if ( v14 != v7 )
          goto LABEL_12;
        v20 = *(_QWORD *)(a1 + 8);
        v21 = *(_QWORD *)(v16 + 8);
        v22 = v20;
        if ( v21 <= v20 )
          v22 = *(_QWORD *)(v16 + 8);
        if ( !v22 )
          break;
        v69 = v19;
        v23 = memcmp(*(const void **)a1, *(const void **)v16, v22);
        v19 = v69;
        if ( !v23 )
          break;
        if ( v23 >= 0 )
        {
LABEL_12:
          v15 = v19;
          if ( v19 >= v16 )
            goto LABEL_26;
LABEL_13:
          v17 = *(_QWORD *)v19;
          v18 = *(_QWORD *)(v19 + 8);
          *(__m128i *)v19 = _mm_loadu_si128((const __m128i *)v16);
          *(_QWORD *)v16 = v17;
          LODWORD(v17) = *(_DWORD *)(v16 + 16);
          *(_QWORD *)(v16 + 8) = v18;
          LODWORD(v18) = *(_DWORD *)(v19 + 16);
          *(_DWORD *)(v19 + 16) = v17;
          v14 = *(_DWORD *)(v16 - 8);
          *(_DWORD *)(v16 + 16) = v18;
          v7 = *(_DWORD *)(a1 + 16);
          goto LABEL_14;
        }
        v16 -= 24LL;
        v14 = *(_DWORD *)(v16 + 16);
      }
      if ( v21 <= v20 )
        break;
LABEL_19:
      v16 -= 24LL;
      v14 = *(_DWORD *)(v16 + 16);
    }
    v15 = v19;
    if ( v19 < v16 )
      goto LABEL_13;
LABEL_26:
    sub_31D6C00(v19, v65, v64);
    result = v15 - a1;
    if ( v15 - a1 > 384 )
    {
      if ( v64 )
      {
        v65 = v15;
        continue;
      }
LABEL_50:
      v35 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
      v36 = (v35 - 2) & 0xFFFFFFFFFFFFFFFELL;
      v37 = (v35 - 2) >> 1;
      v74 = _mm_loadu_si128((const __m128i *)(a1 + 8 * (v37 + v36)));
      sub_31D5B80(
        a1,
        v37,
        v35,
        a4,
        a5,
        a6,
        (const void *)v74.m128i_i64[0],
        v74.m128i_u64[1],
        *(_QWORD *)(a1 + 8 * (v37 + v36) + 16));
      do
      {
        --v37;
        v75 = _mm_loadu_si128((const __m128i *)(a1 + 24 * v37));
        sub_31D5B80(
          a1,
          v37,
          v35,
          v38,
          v39,
          v40,
          (const void *)v75.m128i_i64[0],
          v75.m128i_u64[1],
          *(_QWORD *)(a1 + 24 * v37 + 16));
      }
      while ( v37 );
      v41 = v67;
      do
      {
        v41 = (__m128i *)((char *)v41 - 24);
        v42 = v41[1].m128i_i64[0];
        v43 = _mm_loadu_si128(v41);
        *v41 = _mm_loadu_si128((const __m128i *)a1);
        v41[1].m128i_i32[0] = *(_DWORD *)(a1 + 16);
        result = (__int64)sub_31D5B80(
                            a1,
                            0,
                            0xAAAAAAAAAAAAAAABLL * (((__int64)v41->m128i_i64 - a1) >> 3),
                            v38,
                            v39,
                            v40,
                            (const void *)v43.m128i_i64[0],
                            v43.m128i_u64[1],
                            v42);
      }
      while ( (__int64)v41->m128i_i64 - a1 > 24 );
    }
    return result;
  }
}
