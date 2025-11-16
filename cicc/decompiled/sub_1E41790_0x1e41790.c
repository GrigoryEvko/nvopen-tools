// Function: sub_1E41790
// Address: 0x1e41790
//
__int64 __fastcall sub_1E41790(_QWORD *a1, __int64 *a2)
{
  unsigned __int64 v3; // r12
  const __m128i *v4; // rax
  const __m128i *v5; // rdi
  char *v6; // r13
  unsigned __int64 v7; // rdx
  int *v8; // r15
  int *v9; // r14
  int v10; // r12d
  __int64 v11; // r10
  int v12; // r8d
  int v13; // esi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 i; // r9
  int v17; // ecx
  int v18; // ecx
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // r11
  int *v22; // r14
  int *v23; // r15
  __int64 v24; // r10
  int v25; // edi
  int v26; // r8d
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r9
  int v30; // ecx
  unsigned __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // r11
  __int64 v34; // r14
  __int64 result; // rax
  _QWORD *v36; // r12
  __int64 *v37; // rbx
  __int64 *k; // r15
  __int64 v39; // r13
  _DWORD *v40; // rdx
  int v41; // eax
  const __m128i *v42; // rdi
  unsigned __int64 v43; // r14
  const __m128i *v44; // rdx
  __int64 v45; // rsi
  bool v46; // cf
  unsigned __int64 v47; // rdx
  const __m128i *v48; // rcx
  __m128i *v49; // r15
  __m128i *v50; // r8
  char *v51; // rdx
  __m128i *v52; // rdx
  __m128i *v53; // rax
  __int64 v54; // r15
  __int64 v55; // rax
  __m128i *v56; // [rsp+0h] [rbp-40h]
  __int64 j; // [rsp+8h] [rbp-38h]

  v3 = 0xF0F0F0F0F0F0F0F1LL * ((__int64)(a1[7] - a1[6]) >> 4);
  v4 = (const __m128i *)a1[280];
  v5 = (const __m128i *)a1[279];
  v6 = (char *)((char *)v4 - (char *)v5);
  v7 = v4 - v5;
  if ( v3 <= v7 )
  {
    if ( v3 < v7 )
    {
      v42 = &v5[v3];
      if ( v4 != v42 )
        a1[280] = v42;
    }
    goto LABEL_3;
  }
  v43 = v3 - v7;
  if ( v3 - v7 <= (__int64)(a1[281] - (_QWORD)v4) >> 4 )
  {
    v44 = &v4[v43];
    do
    {
      if ( v4 )
      {
        v4->m128i_i32[0] = 0;
        v4->m128i_i32[1] = 0;
        v4->m128i_i32[2] = 0;
        v4->m128i_i32[3] = 0;
      }
      ++v4;
    }
    while ( v4 != v44 );
    a1[280] = v4;
    goto LABEL_3;
  }
  if ( v43 > 0x7FFFFFFFFFFFFFFLL - v7 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v45 = v3 - v7;
  if ( v7 >= v43 )
    v45 = v4 - v5;
  v46 = __CFADD__(v45, v7);
  v47 = v45 + v7;
  if ( v46 )
  {
    v54 = 0x7FFFFFFFFFFFFFFLL;
  }
  else
  {
    if ( !v47 )
    {
      v48 = v5;
      v49 = 0;
      v50 = 0;
      goto LABEL_59;
    }
    if ( v47 > 0x7FFFFFFFFFFFFFFLL )
      v47 = 0x7FFFFFFFFFFFFFFLL;
    v54 = v47;
  }
  v55 = sub_22077B0(v54 * 16);
  v5 = (const __m128i *)a1[279];
  v50 = (__m128i *)v55;
  v4 = (const __m128i *)a1[280];
  v48 = v5;
  v49 = &v50[v54];
LABEL_59:
  v51 = &v6[(_QWORD)v50];
  do
  {
    if ( v51 )
    {
      *(_DWORD *)v51 = 0;
      *((_DWORD *)v51 + 1) = 0;
      *((_DWORD *)v51 + 2) = 0;
      *((_DWORD *)v51 + 3) = 0;
    }
    v51 += 16;
  }
  while ( v51 != &v6[16 * v43 + (_QWORD)v50] );
  if ( v4 != v5 )
  {
    v52 = (__m128i *)((char *)v50 + (char *)v4 - (char *)v5);
    v53 = v50;
    do
    {
      if ( v53 )
        *v53 = _mm_loadu_si128(v48);
      ++v53;
      ++v48;
    }
    while ( v53 != v52 );
  }
  if ( v5 )
  {
    v56 = v50;
    j_j___libc_free_0(v5, a1[281] - (_QWORD)v5);
    v50 = v56;
  }
  a1[279] = v50;
  a1[281] = v49;
  a1[280] = &v50[v3];
LABEL_3:
  v8 = (int *)a1[271];
  v9 = (int *)a1[270];
  v10 = 0;
  if ( v9 == v8 )
    goto LABEL_33;
  do
  {
    v11 = a1[279];
    v12 = 0;
    v13 = 0;
    v14 = a1[6] + 272LL * *v9;
    v15 = *(_QWORD *)(v14 + 32);
    for ( i = v15 + 16LL * *(unsigned int *)(v14 + 40); i != v15; v15 += 16 )
    {
      while ( 1 )
      {
        v18 = *(_DWORD *)(v15 + 12);
        v19 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v18 && v12 < *(_DWORD *)(v11 + 16LL * *(unsigned int *)(v19 + 192) + 8) + 1 )
          v12 = *(_DWORD *)(v11 + 16LL * *(unsigned int *)(v19 + 192) + 8) + 1;
        v20 = (*(__int64 *)v15 >> 1) & 3;
        if ( v20 == 3 )
          break;
        if ( v20 != 1 )
          goto LABEL_7;
LABEL_9:
        v15 += 16;
        if ( i == v15 )
          goto LABEL_16;
      }
      if ( *(_DWORD *)(v15 + 8) != 3 )
      {
LABEL_7:
        v17 = *(_DWORD *)(v11 + 16LL * *(unsigned int *)(v19 + 192)) + v18;
        if ( v13 < v17 )
          v13 = v17;
        goto LABEL_9;
      }
    }
LABEL_16:
    if ( v10 < v13 )
      v10 = v13;
    v21 = 16LL * *v9++;
    *(_DWORD *)(v11 + v21) = v13;
    *(_DWORD *)(a1[279] + 16LL * *(v9 - 1) + 8) = v12;
  }
  while ( v8 != v9 );
  v22 = (int *)a1[271];
  v23 = (int *)a1[270];
  if ( v23 != v22 )
  {
    while ( 1 )
    {
      v24 = a1[279];
      v25 = v10;
      v26 = 0;
      v27 = a1[6] + 272LL * *(v22 - 1);
      v28 = *(_QWORD *)(v27 + 112);
      v29 = v28 + 16LL * *(unsigned int *)(v27 + 120);
      if ( v28 != v29 )
        break;
LABEL_32:
      v33 = 16LL * *--v22;
      *(_DWORD *)(v24 + v33 + 4) = v25;
      *(_DWORD *)(a1[279] + 16LL * *v22 + 12) = v26;
      if ( v23 == v22 )
        goto LABEL_33;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v30 = *(_DWORD *)(v28 + 12);
        v31 = *(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v30 && v26 < *(_DWORD *)(v24 + 16LL * *(unsigned int *)(v31 + 192) + 12) + 1 )
          v26 = *(_DWORD *)(v24 + 16LL * *(unsigned int *)(v31 + 192) + 12) + 1;
        v32 = (*(__int64 *)v28 >> 1) & 3;
        if ( v32 == 3 )
          break;
        if ( v32 != 1 )
          goto LABEL_23;
LABEL_25:
        v28 += 16;
        if ( v29 == v28 )
          goto LABEL_32;
      }
      if ( *(_DWORD *)(v28 + 8) != 3 )
      {
LABEL_23:
        if ( v25 > *(_DWORD *)(v24 + 16LL * *(unsigned int *)(v31 + 192) + 4) - v30 )
          v25 = *(_DWORD *)(v24 + 16LL * *(unsigned int *)(v31 + 192) + 4) - v30;
        goto LABEL_25;
      }
      v28 += 16;
      if ( v29 == v28 )
        goto LABEL_32;
    }
  }
LABEL_33:
  v34 = *a2;
  result = *a2 + 96LL * *((unsigned int *)a2 + 2);
  v36 = a1;
  for ( j = result; v34 != j; v34 += 96 )
  {
    v37 = *(__int64 **)(v34 + 40);
    for ( k = *(__int64 **)(v34 + 32); v37 != k; *(_DWORD *)(v34 + 68) = result )
    {
      v39 = *k;
      v40 = (_DWORD *)(v36[279] + 16LL * *(unsigned int *)(*k + 192));
      v41 = v40[1] - *v40;
      if ( v41 < *(_DWORD *)(v34 + 64) )
        v41 = *(_DWORD *)(v34 + 64);
      *(_DWORD *)(v34 + 64) = v41;
      if ( (*(_BYTE *)(v39 + 236) & 1) == 0 )
        sub_1F01DD0(v39);
      result = *(unsigned int *)(v34 + 68);
      if ( *(_DWORD *)(v39 + 240) >= (unsigned int)result )
        result = *(unsigned int *)(v39 + 240);
      ++k;
    }
  }
  return result;
}
