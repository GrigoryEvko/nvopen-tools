// Function: sub_11C1A30
// Address: 0x11c1a30
//
__int64 __fastcall sub_11C1A30(const __m128i *a1, const __m128i *a2, __int64 *a3)
{
  __int64 v5; // r12
  __int32 v6; // ebx
  const __m128i *v7; // r8
  int v8; // r15d
  unsigned int v9; // eax
  __int8 *v10; // rdi
  int v11; // r10d
  unsigned __int64 v12; // rax
  unsigned int i; // eax
  __int64 *v14; // rsi
  __int64 v15; // r9
  unsigned int v16; // eax
  unsigned int v17; // r15d
  unsigned __int32 v18; // eax
  int v19; // r8d
  unsigned int v20; // r10d
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  __int32 v23; // eax
  __int64 v24; // rcx
  __m128i *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v29; // r8
  unsigned __int64 v30; // rsi
  __int64 v31; // rax
  __m128i *v32; // rax
  __m128i *v33; // rdx
  const __m128i *v34; // r15
  int v35; // edx
  unsigned int v36; // eax
  int v37; // r9d
  __int8 *v38; // r8
  int j; // eax
  __int64 v40; // rsi
  int v41; // eax
  const __m128i *v42; // r15
  int v43; // edx
  unsigned int v44; // eax
  int v45; // r9d
  int k; // eax
  __int64 v47; // rsi
  int v48; // eax
  __int32 v49; // eax
  __int32 v50; // eax
  __int64 m128i_i64; // rdi
  __int64 v52; // r9
  __int8 *v53; // rbx
  char v54; // [rsp+Fh] [rbp-61h]
  const __m128i *v56; // [rsp+18h] [rbp-58h]
  int v57; // [rsp+18h] [rbp-58h]
  int v58; // [rsp+18h] [rbp-58h]
  __m128i v59; // [rsp+20h] [rbp-50h] BYREF
  __int64 v60; // [rsp+30h] [rbp-40h]

  v5 = a2->m128i_i64[0];
  v6 = a2->m128i_i32[2];
  v54 = a1->m128i_i8[8] & 1;
  if ( v54 )
  {
    v7 = a1 + 1;
    v8 = 7;
  }
  else
  {
    v17 = a1[1].m128i_u32[2];
    v7 = (const __m128i *)a1[1].m128i_i64[0];
    if ( !v17 )
    {
      v18 = a1->m128i_u32[2];
      ++a1->m128i_i64[0];
      v10 = 0;
      v19 = (v18 >> 1) + 1;
LABEL_14:
      v20 = 3 * v17;
      goto LABEL_15;
    }
    v8 = v17 - 1;
  }
  v56 = v7;
  v59.m128i_i32[0] = a2->m128i_i32[2];
  v9 = sub_CF97C0((unsigned int *)&v59);
  v10 = 0;
  v11 = 1;
  v12 = 0xBF58476D1CE4E5B9LL * (v9 | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32));
  for ( i = v8 & ((v12 >> 31) ^ v12); ; i = v8 & v16 )
  {
    v14 = &v56->m128i_i64[3 * i];
    v15 = *v14;
    if ( v5 == *v14 && *((_DWORD *)v14 + 2) == v6 )
      return a1[13].m128i_i64[0] + 24LL * *((unsigned int *)v14 + 4);
    if ( v15 == -4096 )
      break;
    if ( v15 == -8192 && *((_DWORD *)v14 + 2) == 101 && !v10 )
      v10 = &v56->m128i_i8[24 * i];
LABEL_10:
    v16 = v11 + i;
    ++v11;
  }
  if ( *((_DWORD *)v14 + 2) != 100 )
    goto LABEL_10;
  v18 = a1->m128i_u32[2];
  if ( !v10 )
    v10 = (__int8 *)v14;
  ++a1->m128i_i64[0];
  v19 = (v18 >> 1) + 1;
  if ( !v54 )
  {
    v17 = a1[1].m128i_u32[2];
    goto LABEL_14;
  }
  v20 = 24;
  v17 = 8;
LABEL_15:
  if ( 4 * v19 >= v20 )
  {
    sub_11C13E0(a1, 2 * v17);
    if ( (a1->m128i_i8[8] & 1) != 0 )
    {
      v34 = a1 + 1;
      v35 = 7;
    }
    else
    {
      v49 = a1[1].m128i_i32[2];
      v34 = (const __m128i *)a1[1].m128i_i64[0];
      if ( !v49 )
        goto LABEL_73;
      v35 = v49 - 1;
    }
    v57 = v35;
    v59.m128i_i32[0] = v6;
    v36 = sub_CF97C0((unsigned int *)&v59);
    v37 = 1;
    v38 = 0;
    for ( j = v57
            & (((0xBF58476D1CE4E5B9LL
               * (v36 | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32))) >> 31)
             ^ (484763065 * v36)); ; j = v57 & v41 )
    {
      v10 = &v34->m128i_i8[24 * j];
      v40 = *(_QWORD *)v10;
      if ( v5 == *(_QWORD *)v10 && v6 == *((_DWORD *)v10 + 2) )
        break;
      if ( v40 == -4096 )
      {
        if ( *((_DWORD *)v10 + 2) == 100 )
        {
LABEL_68:
          if ( v38 )
            v10 = v38;
          goto LABEL_60;
        }
      }
      else if ( v40 == -8192 && *((_DWORD *)v10 + 2) == 101 && !v38 )
      {
        v38 = &v34->m128i_i8[24 * j];
      }
      v41 = v37 + j;
      ++v37;
    }
    goto LABEL_60;
  }
  if ( v17 - a1->m128i_i32[3] - v19 <= v17 >> 3 )
  {
    sub_11C13E0(a1, v17);
    if ( (a1->m128i_i8[8] & 1) != 0 )
    {
      v42 = a1 + 1;
      v43 = 7;
      goto LABEL_47;
    }
    v50 = a1[1].m128i_i32[2];
    v42 = (const __m128i *)a1[1].m128i_i64[0];
    if ( v50 )
    {
      v43 = v50 - 1;
LABEL_47:
      v58 = v43;
      v59.m128i_i32[0] = v6;
      v44 = sub_CF97C0((unsigned int *)&v59);
      v45 = 1;
      v38 = 0;
      for ( k = v58
              & (((0xBF58476D1CE4E5B9LL
                 * (v44 | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32))) >> 31)
               ^ (484763065 * v44)); ; k = v58 & v48 )
      {
        v10 = &v42->m128i_i8[24 * k];
        v47 = *(_QWORD *)v10;
        if ( v5 == *(_QWORD *)v10 && v6 == *((_DWORD *)v10 + 2) )
          break;
        if ( v47 == -4096 )
        {
          if ( *((_DWORD *)v10 + 2) == 100 )
            goto LABEL_68;
        }
        else if ( v47 == -8192 && *((_DWORD *)v10 + 2) == 101 && !v38 )
        {
          v38 = &v42->m128i_i8[24 * k];
        }
        v48 = v45 + k;
        ++v45;
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
  if ( *(_QWORD *)v10 != -4096 || *((_DWORD *)v10 + 2) != 100 )
    --a1->m128i_i32[3];
  *((_DWORD *)v10 + 4) = 0;
  *(_QWORD *)v10 = v5;
  *((_DWORD *)v10 + 2) = v6;
  *((_DWORD *)v10 + 4) = a1[13].m128i_i32[2];
  v21 = a1[13].m128i_u32[2];
  v22 = a1[13].m128i_u32[3];
  v23 = a1[13].m128i_i32[2];
  if ( v21 >= v22 )
  {
    v29 = v21 + 1;
    v30 = a1[13].m128i_u64[0];
    v31 = *a3;
    v59 = _mm_loadu_si128(a2);
    v60 = v31;
    v32 = &v59;
    if ( v22 < v21 + 1 )
    {
      m128i_i64 = (__int64)a1[13].m128i_i64;
      v52 = (__int64)a1[14].m128i_i64;
      if ( v30 > (unsigned __int64)&v59 || (unsigned __int64)&v59 >= v30 + 24 * v21 )
      {
        sub_C8D5F0(m128i_i64, &a1[14], v29, 0x18u, v29, v52);
        v30 = a1[13].m128i_u64[0];
        v21 = a1[13].m128i_u32[2];
        v32 = &v59;
      }
      else
      {
        v53 = &v59.m128i_i8[-v30];
        sub_C8D5F0(m128i_i64, &a1[14], v29, 0x18u, v29, v52);
        v30 = a1[13].m128i_u64[0];
        v21 = a1[13].m128i_u32[2];
        v32 = (__m128i *)&v53[v30];
      }
    }
    v33 = (__m128i *)(v30 + 24 * v21);
    *v33 = _mm_loadu_si128(v32);
    v33[1].m128i_i64[0] = v32[1].m128i_i64[0];
    v24 = a1[13].m128i_i64[0];
    v27 = (unsigned int)(a1[13].m128i_i32[2] + 1);
    a1[13].m128i_i32[2] = v27;
  }
  else
  {
    v24 = a1[13].m128i_i64[0];
    v25 = (__m128i *)(v24 + 24 * v21);
    if ( v25 )
    {
      v26 = *a3;
      *v25 = _mm_loadu_si128(a2);
      v25[1].m128i_i64[0] = v26;
      v23 = a1[13].m128i_i32[2];
      v24 = a1[13].m128i_i64[0];
    }
    v27 = (unsigned int)(v23 + 1);
    a1[13].m128i_i32[2] = v27;
  }
  return v24 + 24 * v27 - 24;
}
