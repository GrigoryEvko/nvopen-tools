// Function: sub_9F2550
// Address: 0x9f2550
//
__int64 __fastcall sub_9F2550(const __m128i *a1, const __m128i *a2, __int64 *a3)
{
  __int64 v6; // r13
  __int64 v7; // r15
  char v8; // dl
  const __m128i *v9; // rcx
  int v10; // esi
  int v11; // r10d
  __int8 *v12; // rdi
  int i; // eax
  __int8 *v14; // r8
  __int64 v15; // r9
  int v16; // eax
  unsigned int v17; // esi
  unsigned __int32 v18; // eax
  int v19; // ecx
  unsigned int v20; // r8d
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  __int32 v23; // eax
  __int64 v24; // rcx
  __m128i *v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // rax
  __m128i v29; // xmm1
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // rsi
  const __m128i *v32; // rax
  __m128i *v33; // rdx
  const __m128i *v34; // rcx
  int v35; // edx
  int v36; // r9d
  __int8 *v37; // r8
  int j; // eax
  __int64 v39; // rsi
  int v40; // eax
  const __m128i *v41; // rcx
  int v42; // edx
  int v43; // r9d
  int k; // eax
  __int64 v45; // rsi
  int v46; // eax
  __int32 v47; // edx
  __int32 v48; // edx
  const __m128i *v49; // rdi
  __int8 *v50; // r12
  __m128i v51; // [rsp+0h] [rbp-50h] BYREF
  __int64 v52; // [rsp+10h] [rbp-40h]

  v6 = a2->m128i_i64[0];
  v7 = a2->m128i_i64[1];
  v8 = a1->m128i_i8[8] & 1;
  if ( v8 )
  {
    v9 = a1 + 1;
    v10 = 3;
  }
  else
  {
    v17 = a1[1].m128i_u32[2];
    v9 = (const __m128i *)a1[1].m128i_i64[0];
    if ( !v17 )
    {
      v18 = a1->m128i_u32[2];
      ++a1->m128i_i64[0];
      v12 = 0;
      v19 = (v18 >> 1) + 1;
LABEL_14:
      v20 = 3 * v17;
      goto LABEL_15;
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
    v14 = &v9->m128i_i8[24 * i];
    v15 = *(_QWORD *)v14;
    if ( v6 == *(_QWORD *)v14 && *((_QWORD *)v14 + 1) == v7 )
      return a1[7].m128i_i64[0] + 24LL * *((unsigned int *)v14 + 4);
    if ( v15 == -4096 )
      break;
    if ( v15 == -8192 && *((_QWORD *)v14 + 1) == -8192 && !v12 )
      v12 = &v9->m128i_i8[24 * i];
LABEL_10:
    v16 = v11 + i;
    ++v11;
  }
  if ( *((_QWORD *)v14 + 1) != -4096 )
    goto LABEL_10;
  v18 = a1->m128i_u32[2];
  if ( !v12 )
    v12 = v14;
  ++a1->m128i_i64[0];
  v19 = (v18 >> 1) + 1;
  if ( !v8 )
  {
    v17 = a1[1].m128i_u32[2];
    goto LABEL_14;
  }
  v20 = 12;
  v17 = 4;
LABEL_15:
  if ( v20 <= 4 * v19 )
  {
    sub_9F1FB0(a1, 2 * v17);
    if ( (a1->m128i_i8[8] & 1) != 0 )
    {
      v34 = a1 + 1;
      v35 = 3;
    }
    else
    {
      v47 = a1[1].m128i_i32[2];
      v34 = (const __m128i *)a1[1].m128i_i64[0];
      if ( !v47 )
        goto LABEL_73;
      v35 = v47 - 1;
    }
    v36 = 1;
    v37 = 0;
    for ( j = v35
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
                | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))); ; j = v35 & v40 )
    {
      v12 = &v34->m128i_i8[24 * j];
      v39 = *(_QWORD *)v12;
      if ( v6 == *(_QWORD *)v12 && v7 == *((_QWORD *)v12 + 1) )
        break;
      if ( v39 == -4096 )
      {
        if ( *((_QWORD *)v12 + 1) == -4096 )
        {
LABEL_68:
          if ( v37 )
            v12 = v37;
          goto LABEL_60;
        }
      }
      else if ( v39 == -8192 && *((_QWORD *)v12 + 1) == -8192 && !v37 )
      {
        v37 = &v34->m128i_i8[24 * j];
      }
      v40 = v36 + j;
      ++v36;
    }
    goto LABEL_60;
  }
  if ( v17 - a1->m128i_i32[3] - v19 <= v17 >> 3 )
  {
    sub_9F1FB0(a1, v17);
    if ( (a1->m128i_i8[8] & 1) != 0 )
    {
      v41 = a1 + 1;
      v42 = 3;
      goto LABEL_47;
    }
    v48 = a1[1].m128i_i32[2];
    v41 = (const __m128i *)a1[1].m128i_i64[0];
    if ( v48 )
    {
      v42 = v48 - 1;
LABEL_47:
      v43 = 1;
      v37 = 0;
      for ( k = v42
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
                  | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))); ; k = v42 & v46 )
      {
        v12 = &v41->m128i_i8[24 * k];
        v45 = *(_QWORD *)v12;
        if ( v6 == *(_QWORD *)v12 && v7 == *((_QWORD *)v12 + 1) )
          break;
        if ( v45 == -4096 )
        {
          if ( *((_QWORD *)v12 + 1) == -4096 )
            goto LABEL_68;
        }
        else if ( v45 == -8192 && *((_QWORD *)v12 + 1) == -8192 && !v37 )
        {
          v37 = &v41->m128i_i8[24 * k];
        }
        v46 = v43 + k;
        ++v43;
      }
LABEL_60:
      v18 = a1->m128i_u32[2];
      goto LABEL_17;
    }
LABEL_73:
    a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    BUG();
  }
LABEL_17:
  a1->m128i_i32[2] = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( *(_QWORD *)v12 != -4096 || *((_QWORD *)v12 + 1) != -4096 )
    --a1->m128i_i32[3];
  *((_DWORD *)v12 + 4) = 0;
  *(_QWORD *)v12 = v6;
  *((_QWORD *)v12 + 1) = v7;
  *((_DWORD *)v12 + 4) = a1[7].m128i_i32[2];
  v21 = a1[7].m128i_u32[2];
  v22 = a1[7].m128i_u32[3];
  v23 = a1[7].m128i_i32[2];
  if ( v21 >= v22 )
  {
    v29 = _mm_loadu_si128(a2);
    v30 = v21 + 1;
    v31 = a1[7].m128i_u64[0];
    v52 = *a3;
    v32 = &v51;
    v51 = v29;
    if ( v22 < v21 + 1 )
    {
      v49 = a1 + 7;
      if ( v31 > (unsigned __int64)&v51 || (unsigned __int64)&v51 >= v31 + 24 * v21 )
      {
        sub_C8D5F0(v49, &a1[8], v30, 24);
        v31 = a1[7].m128i_u64[0];
        v21 = a1[7].m128i_u32[2];
        v32 = &v51;
      }
      else
      {
        v50 = &v51.m128i_i8[-v31];
        sub_C8D5F0(v49, &a1[8], v30, 24);
        v31 = a1[7].m128i_u64[0];
        v21 = a1[7].m128i_u32[2];
        v32 = (const __m128i *)&v50[v31];
      }
    }
    v33 = (__m128i *)(v31 + 24 * v21);
    *v33 = _mm_loadu_si128(v32);
    v33[1].m128i_i64[0] = v32[1].m128i_i64[0];
    v24 = a1[7].m128i_i64[0];
    v27 = (unsigned int)(a1[7].m128i_i32[2] + 1);
    a1[7].m128i_i32[2] = v27;
  }
  else
  {
    v24 = a1[7].m128i_i64[0];
    v25 = (__m128i *)(v24 + 24 * v21);
    if ( v25 )
    {
      v26 = _mm_loadu_si128(a2);
      v25[1].m128i_i64[0] = *a3;
      *v25 = v26;
      v23 = a1[7].m128i_i32[2];
      v24 = a1[7].m128i_i64[0];
    }
    v27 = (unsigned int)(v23 + 1);
    a1[7].m128i_i32[2] = v27;
  }
  return v24 + 24 * v27 - 24;
}
