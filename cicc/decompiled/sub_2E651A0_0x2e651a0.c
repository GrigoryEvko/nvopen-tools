// Function: sub_2E651A0
// Address: 0x2e651a0
//
bool __fastcall sub_2E651A0(const __m128i **a1, __int64 *a2, __int64 *a3)
{
  const __m128i *v5; // rbx
  __int64 v6; // r14
  unsigned __int64 v7; // r15
  char v8; // dl
  const __m128i *v9; // rcx
  int v10; // esi
  int v11; // r10d
  __int64 *v12; // rdi
  unsigned int i; // eax
  __int64 *v14; // r8
  __int64 v15; // r9
  unsigned int v16; // eax
  unsigned int v17; // esi
  unsigned __int32 v18; // eax
  int v19; // ecx
  unsigned int v20; // r8d
  int *v21; // r14
  __int64 v22; // r15
  unsigned __int64 v23; // r13
  const __m128i *v24; // rcx
  int v25; // esi
  int v26; // r10d
  __int8 *v27; // rdi
  int v28; // eax
  __int8 *v29; // r8
  __int64 v30; // r9
  int v31; // eax
  unsigned int v32; // esi
  int v33; // ecx
  unsigned __int32 v34; // eax
  int v35; // ecx
  unsigned int v36; // r8d
  int v37; // eax
  const __m128i *v39; // rcx
  int v40; // edx
  int v41; // r9d
  __int8 *v42; // r8
  unsigned int j; // eax
  __int64 v44; // rsi
  unsigned int v45; // eax
  const __m128i *v46; // rcx
  int v47; // edx
  int v48; // r9d
  __int8 *v49; // r8
  int m; // eax
  __int64 v51; // rsi
  int v52; // eax
  const __m128i *v53; // rcx
  int v54; // edx
  int v55; // r9d
  unsigned int k; // eax
  __int64 v57; // rsi
  unsigned int v58; // eax
  const __m128i *v59; // rcx
  int v60; // edx
  int v61; // r9d
  int n; // eax
  __int64 v63; // rsi
  int v64; // eax
  __int32 v65; // edx
  __int32 v66; // edx
  __int32 v67; // edx
  __int32 v68; // edx

  v5 = *a1;
  v6 = *a2;
  v7 = a2[1] & 0xFFFFFFFFFFFFFFF8LL;
  v8 = (*a1)->m128i_i8[8] & 1;
  if ( v8 )
  {
    v9 = v5 + 1;
    v10 = 3;
  }
  else
  {
    v17 = v5[1].m128i_u32[2];
    v9 = (const __m128i *)v5[1].m128i_i64[0];
    if ( !v17 )
    {
      v18 = v5->m128i_u32[2];
      ++v5->m128i_i64[0];
      v12 = 0;
      v19 = (v18 >> 1) + 1;
      goto LABEL_14;
    }
    v10 = v17 - 1;
  }
  v11 = 1;
  v12 = 0;
  for ( i = v10
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
              | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))); ; i = v10 & v16 )
  {
    v14 = &v9->m128i_i64[3 * i];
    v15 = *v14;
    if ( v6 == *v14 && v7 == v14[1] )
    {
      v21 = (int *)(v14 + 2);
      goto LABEL_20;
    }
    if ( v15 == -4096 )
      break;
    if ( v15 == -8192 && v14[1] == -8192 && !v12 )
      v12 = &v9->m128i_i64[3 * i];
LABEL_10:
    v16 = v11 + i;
    ++v11;
  }
  if ( v14[1] != -4096 )
    goto LABEL_10;
  v18 = v5->m128i_u32[2];
  v17 = 4;
  if ( !v12 )
    v12 = v14;
  ++v5->m128i_i64[0];
  v20 = 12;
  v19 = (v18 >> 1) + 1;
  if ( !v8 )
  {
    v17 = v5[1].m128i_u32[2];
LABEL_14:
    v20 = 3 * v17;
  }
  if ( 4 * v19 >= v20 )
  {
    sub_2E64C00(v5, 2 * v17);
    if ( (v5->m128i_i8[8] & 1) != 0 )
    {
      v39 = v5 + 1;
      v40 = 3;
    }
    else
    {
      v65 = v5[1].m128i_i32[2];
      v39 = (const __m128i *)v5[1].m128i_i64[0];
      if ( !v65 )
        goto LABEL_126;
      v40 = v65 - 1;
    }
    v41 = 1;
    v42 = 0;
    for ( j = v40
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
                | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))); ; j = v40 & v45 )
    {
      v12 = &v39->m128i_i64[3 * j];
      v44 = *v12;
      if ( v6 == *v12 && v7 == v12[1] )
        break;
      if ( v44 == -4096 )
      {
        if ( v12[1] == -4096 )
        {
LABEL_117:
          if ( v42 )
            v12 = (__int64 *)v42;
          goto LABEL_111;
        }
      }
      else if ( v44 == -8192 && v12[1] == -8192 && !v42 )
      {
        v42 = &v39->m128i_i8[24 * j];
      }
      v45 = v41 + j;
      ++v41;
    }
    goto LABEL_111;
  }
  if ( v17 - v5->m128i_i32[3] - v19 <= v17 >> 3 )
  {
    sub_2E64C00(v5, v17);
    if ( (v5->m128i_i8[8] & 1) != 0 )
    {
      v53 = v5 + 1;
      v54 = 3;
    }
    else
    {
      v68 = v5[1].m128i_i32[2];
      v53 = (const __m128i *)v5[1].m128i_i64[0];
      if ( !v68 )
        goto LABEL_126;
      v54 = v68 - 1;
    }
    v55 = 1;
    v42 = 0;
    for ( k = v54
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
                | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))); ; k = v54 & v58 )
    {
      v12 = &v53->m128i_i64[3 * k];
      v57 = *v12;
      if ( v6 == *v12 && v7 == v12[1] )
        break;
      if ( v57 == -4096 )
      {
        if ( v12[1] == -4096 )
          goto LABEL_117;
      }
      else if ( v57 == -8192 && v12[1] == -8192 && !v42 )
      {
        v42 = &v53->m128i_i8[24 * k];
      }
      v58 = v55 + k;
      ++v55;
    }
LABEL_111:
    v18 = v5->m128i_u32[2];
  }
  v5->m128i_i32[2] = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( *v12 != -4096 || v12[1] != -4096 )
    --v5->m128i_i32[3];
  *v12 = v6;
  v21 = (int *)(v12 + 2);
  v12[1] = v7;
  *((_DWORD *)v12 + 4) = 0;
  v5 = *a1;
  v8 = (*a1)->m128i_i8[8] & 1;
LABEL_20:
  v22 = *a3;
  v23 = a3[1] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 )
  {
    v24 = v5 + 1;
    v25 = 3;
  }
  else
  {
    v32 = v5[1].m128i_u32[2];
    v24 = (const __m128i *)v5[1].m128i_i64[0];
    if ( !v32 )
    {
      v34 = v5->m128i_u32[2];
      ++v5->m128i_i64[0];
      v27 = 0;
      v35 = (v34 >> 1) + 1;
      goto LABEL_37;
    }
    v25 = v32 - 1;
  }
  v26 = 1;
  v27 = 0;
  v28 = v25
      & (((0xBF58476D1CE4E5B9LL
         * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)
          | ((unsigned __int64)(((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))) >> 31)
       ^ (484763065 * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4))));
  while ( 2 )
  {
    v29 = &v24->m128i_i8[24 * v28];
    v30 = *(_QWORD *)v29;
    if ( v22 == *(_QWORD *)v29 && v23 == *((_QWORD *)v29 + 1) )
    {
      v33 = *((_DWORD *)v29 + 4);
      goto LABEL_43;
    }
    if ( v30 != -4096 )
    {
      if ( v30 == -8192 && *((_QWORD *)v29 + 1) == -8192 && !v27 )
        v27 = &v24->m128i_i8[24 * v28];
      goto LABEL_29;
    }
    if ( *((_QWORD *)v29 + 1) != -4096 )
    {
LABEL_29:
      v31 = v26 + v28;
      ++v26;
      v28 = v25 & v31;
      continue;
    }
    break;
  }
  v34 = v5->m128i_u32[2];
  v32 = 4;
  if ( !v27 )
    v27 = v29;
  ++v5->m128i_i64[0];
  v36 = 12;
  v35 = (v34 >> 1) + 1;
  if ( !v8 )
  {
    v32 = v5[1].m128i_u32[2];
LABEL_37:
    v36 = 3 * v32;
  }
  if ( v36 <= 4 * v35 )
  {
    sub_2E64C00(v5, 2 * v32);
    if ( (v5->m128i_i8[8] & 1) != 0 )
    {
      v46 = v5 + 1;
      v47 = 3;
    }
    else
    {
      v66 = v5[1].m128i_i32[2];
      v46 = (const __m128i *)v5[1].m128i_i64[0];
      if ( !v66 )
        goto LABEL_126;
      v47 = v66 - 1;
    }
    v48 = 1;
    v49 = 0;
    for ( m = v47
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)
                | ((unsigned __int64)(((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)))); ; m = v47 & v52 )
    {
      v27 = &v46->m128i_i8[24 * m];
      v51 = *(_QWORD *)v27;
      if ( v22 == *(_QWORD *)v27 && v23 == *((_QWORD *)v27 + 1) )
        break;
      if ( v51 == -4096 )
      {
        if ( *((_QWORD *)v27 + 1) == -4096 )
        {
LABEL_120:
          if ( v49 )
            v27 = v49;
          goto LABEL_109;
        }
      }
      else if ( v51 == -8192 && *((_QWORD *)v27 + 1) == -8192 && !v49 )
      {
        v49 = &v46->m128i_i8[24 * m];
      }
      v52 = v48 + m;
      ++v48;
    }
    goto LABEL_109;
  }
  if ( v32 - v5->m128i_i32[3] - v35 <= v32 >> 3 )
  {
    sub_2E64C00(v5, v32);
    if ( (v5->m128i_i8[8] & 1) != 0 )
    {
      v59 = v5 + 1;
      v60 = 3;
      goto LABEL_92;
    }
    v67 = v5[1].m128i_i32[2];
    v59 = (const __m128i *)v5[1].m128i_i64[0];
    if ( v67 )
    {
      v60 = v67 - 1;
LABEL_92:
      v61 = 1;
      v49 = 0;
      for ( n = v60
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)
                  | ((unsigned __int64)(((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)))); ; n = v60 & v64 )
      {
        v27 = &v59->m128i_i8[24 * n];
        v63 = *(_QWORD *)v27;
        if ( v22 == *(_QWORD *)v27 && v23 == *((_QWORD *)v27 + 1) )
          break;
        if ( v63 == -4096 )
        {
          if ( *((_QWORD *)v27 + 1) == -4096 )
            goto LABEL_120;
        }
        else if ( v63 == -8192 && *((_QWORD *)v27 + 1) == -8192 && !v49 )
        {
          v49 = &v59->m128i_i8[24 * n];
        }
        v64 = v61 + n;
        ++v61;
      }
LABEL_109:
      v34 = v5->m128i_u32[2];
      goto LABEL_40;
    }
LABEL_126:
    v5->m128i_i32[2] = (2 * ((unsigned __int32)v5->m128i_i32[2] >> 1) + 2) | v5->m128i_i32[2] & 1;
    BUG();
  }
LABEL_40:
  v5->m128i_i32[2] = (2 * (v34 >> 1) + 2) | v34 & 1;
  if ( *(_QWORD *)v27 != -4096 || *((_QWORD *)v27 + 1) != -4096 )
    --v5->m128i_i32[3];
  *(_QWORD *)v27 = v22;
  v33 = 0;
  *((_QWORD *)v27 + 1) = v23;
  *((_DWORD *)v27 + 4) = 0;
LABEL_43:
  v37 = *v21;
  if ( a1[1]->m128i_i8[0] )
    return v33 > v37;
  else
    return v33 < v37;
}
