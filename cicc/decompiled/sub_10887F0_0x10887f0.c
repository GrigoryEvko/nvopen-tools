// Function: sub_10887F0
// Address: 0x10887f0
//
__int64 __fastcall sub_10887F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // r15
  _QWORD *v5; // r14
  _QWORD *v6; // r13
  _QWORD *v7; // r12
  _QWORD *v8; // rdi
  __int64 v9; // rdi
  _QWORD *v10; // rdi
  __int64 *v11; // r15
  __int64 *v12; // r14
  __int64 *v13; // r13
  __int64 v14; // r12
  __int64 v15; // rdi
  __int64 v16; // rdi
  int v17; // eax
  __int64 v18; // rdx
  _QWORD *v19; // rax
  _QWORD *i; // rdx
  int v21; // eax
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *k; // rdx
  int v25; // eax
  __int64 result; // rax
  __int64 v27; // rdx
  __int64 n; // rdx
  unsigned int v29; // ecx
  unsigned int v30; // eax
  _QWORD *v31; // rdi
  __int64 v32; // r12
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // rdx
  __int64 ii; // rdx
  unsigned int v37; // ecx
  unsigned int v38; // eax
  _QWORD *v39; // rdi
  int v40; // r12d
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rdi
  _QWORD *v43; // rax
  __int64 v44; // rdx
  _QWORD *m; // rdx
  unsigned int v46; // ecx
  unsigned int v47; // eax
  _QWORD *v48; // rdi
  int v49; // r12d
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rdi
  _QWORD *v52; // rax
  __int64 v53; // rdx
  _QWORD *j; // rdx
  _QWORD *v55; // rax
  _QWORD *v56; // rax

  v3 = *a1;
  *(_OWORD *)(a1 + 3) = 0;
  v4 = (_QWORD *)a1[6];
  a1[5] = 0;
  v5 = (_QWORD *)a1[7];
  *((_WORD *)a1 + 12) = *(_DWORD *)(*(_QWORD *)(v3 + 104) + 8LL);
  if ( v4 != v5 )
  {
    v6 = v4;
    do
    {
      v7 = (_QWORD *)*v6;
      if ( *v6 )
      {
        v8 = (_QWORD *)v7[15];
        if ( v8 != v7 + 17 )
          _libc_free(v8, a2);
        v9 = v7[12];
        if ( v9 )
          j_j___libc_free_0(v9, v7[14] - v9);
        v10 = (_QWORD *)v7[5];
        if ( v10 != v7 + 7 )
          j_j___libc_free_0(v10, v7[7] + 1LL);
        a2 = 144;
        j_j___libc_free_0(v7, 144);
      }
      ++v6;
    }
    while ( v5 != v6 );
    a1[7] = (__int64)v4;
  }
  v11 = (__int64 *)a1[9];
  v12 = (__int64 *)a1[10];
  if ( v11 != v12 )
  {
    v13 = (__int64 *)a1[9];
    do
    {
      v14 = *v13;
      if ( *v13 )
      {
        v15 = *(_QWORD *)(v14 + 64);
        if ( v15 != v14 + 80 )
          _libc_free(v15, a2);
        v16 = *(_QWORD *)(v14 + 24);
        if ( v16 != v14 + 48 )
          _libc_free(v16, a2);
        a2 = 136;
        j_j___libc_free_0(v14, 136);
      }
      ++v13;
    }
    while ( v12 != v13 );
    a1[10] = (__int64)v11;
  }
  sub_C0C1A0((__int64)(a1 + 12));
  v17 = *((_DWORD *)a1 + 40);
  ++a1[18];
  if ( !v17 )
  {
    if ( !*((_DWORD *)a1 + 41) )
      goto LABEL_29;
    v18 = *((unsigned int *)a1 + 42);
    if ( (unsigned int)v18 > 0x40 )
    {
      sub_C7D6A0(a1[19], 16LL * (unsigned int)v18, 8);
      a1[19] = 0;
      a1[20] = 0;
      *((_DWORD *)a1 + 42) = 0;
      goto LABEL_29;
    }
    goto LABEL_26;
  }
  v46 = 4 * v17;
  v18 = *((unsigned int *)a1 + 42);
  if ( (unsigned int)(4 * v17) < 0x40 )
    v46 = 64;
  if ( (unsigned int)v18 <= v46 )
  {
LABEL_26:
    v19 = (_QWORD *)a1[19];
    for ( i = &v19[2 * v18]; i != v19; v19 += 2 )
      *v19 = -4096;
    a1[20] = 0;
    goto LABEL_29;
  }
  v47 = v17 - 1;
  if ( !v47 )
  {
    v48 = (_QWORD *)a1[19];
    v49 = 64;
LABEL_73:
    sub_C7D6A0((__int64)v48, 16LL * (unsigned int)v18, 8);
    v50 = ((((((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
             | (4 * v49 / 3u + 1)
             | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
           | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
           | (4 * v49 / 3u + 1)
           | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
           | (4 * v49 / 3u + 1)
           | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
         | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
         | (4 * v49 / 3u + 1)
         | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 16;
    v51 = (v50
         | (((((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
             | (4 * v49 / 3u + 1)
             | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
           | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
           | (4 * v49 / 3u + 1)
           | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
           | (4 * v49 / 3u + 1)
           | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
         | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
         | (4 * v49 / 3u + 1)
         | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1))
        + 1;
    *((_DWORD *)a1 + 42) = v51;
    v52 = (_QWORD *)sub_C7D670(16 * v51, 8);
    v53 = *((unsigned int *)a1 + 42);
    a1[20] = 0;
    a1[19] = (__int64)v52;
    for ( j = &v52[2 * v53]; j != v52; v52 += 2 )
    {
      if ( v52 )
        *v52 = -4096;
    }
    goto LABEL_29;
  }
  _BitScanReverse(&v47, v47);
  v48 = (_QWORD *)a1[19];
  v49 = 1 << (33 - (v47 ^ 0x1F));
  if ( v49 < 64 )
    v49 = 64;
  if ( (_DWORD)v18 != v49 )
    goto LABEL_73;
  a1[20] = 0;
  v56 = &v48[2 * (unsigned int)v18];
  do
  {
    if ( v48 )
      *v48 = -4096;
    v48 += 2;
  }
  while ( v56 != v48 );
LABEL_29:
  v21 = *((_DWORD *)a1 + 48);
  ++a1[22];
  if ( !v21 )
  {
    if ( !*((_DWORD *)a1 + 49) )
      goto LABEL_35;
    v22 = *((unsigned int *)a1 + 50);
    if ( (unsigned int)v22 > 0x40 )
    {
      sub_C7D6A0(a1[23], 16LL * (unsigned int)v22, 8);
      a1[23] = 0;
      a1[24] = 0;
      *((_DWORD *)a1 + 50) = 0;
      goto LABEL_35;
    }
    goto LABEL_32;
  }
  v37 = 4 * v21;
  v22 = *((unsigned int *)a1 + 50);
  if ( (unsigned int)(4 * v21) < 0x40 )
    v37 = 64;
  if ( v37 >= (unsigned int)v22 )
  {
LABEL_32:
    v23 = (_QWORD *)a1[23];
    for ( k = &v23[2 * v22]; k != v23; v23 += 2 )
      *v23 = -4096;
    a1[24] = 0;
    goto LABEL_35;
  }
  v38 = v21 - 1;
  if ( !v38 )
  {
    v39 = (_QWORD *)a1[23];
    v40 = 64;
LABEL_61:
    sub_C7D6A0((__int64)v39, 16LL * (unsigned int)v22, 8);
    v41 = ((((((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
             | (4 * v40 / 3u + 1)
             | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
           | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
         | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
         | (4 * v40 / 3u + 1)
         | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 16;
    v42 = (v41
         | (((((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
             | (4 * v40 / 3u + 1)
             | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
           | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
         | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
         | (4 * v40 / 3u + 1)
         | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1))
        + 1;
    *((_DWORD *)a1 + 50) = v42;
    v43 = (_QWORD *)sub_C7D670(16 * v42, 8);
    v44 = *((unsigned int *)a1 + 50);
    a1[24] = 0;
    a1[23] = (__int64)v43;
    for ( m = &v43[2 * v44]; m != v43; v43 += 2 )
    {
      if ( v43 )
        *v43 = -4096;
    }
    goto LABEL_35;
  }
  _BitScanReverse(&v38, v38);
  v39 = (_QWORD *)a1[23];
  v40 = 1 << (33 - (v38 ^ 0x1F));
  if ( v40 < 64 )
    v40 = 64;
  if ( (_DWORD)v22 != v40 )
    goto LABEL_61;
  a1[24] = 0;
  v55 = &v39[2 * (unsigned int)v22];
  do
  {
    if ( v39 )
      *v39 = -4096;
    v39 += 2;
  }
  while ( v55 != v39 );
LABEL_35:
  v25 = *((_DWORD *)a1 + 56);
  ++a1[26];
  if ( !v25 )
  {
    result = *((unsigned int *)a1 + 57);
    if ( !(_DWORD)result )
      return result;
    v27 = *((unsigned int *)a1 + 58);
    if ( (unsigned int)v27 > 0x40 )
    {
      result = sub_C7D6A0(a1[27], 8 * v27, 8);
      a1[27] = 0;
      a1[28] = 0;
      *((_DWORD *)a1 + 58) = 0;
      return result;
    }
    goto LABEL_38;
  }
  v29 = 4 * v25;
  v27 = *((unsigned int *)a1 + 58);
  if ( (unsigned int)(4 * v25) < 0x40 )
    v29 = 64;
  if ( v29 >= (unsigned int)v27 )
  {
LABEL_38:
    result = a1[27];
    for ( n = result + 8 * v27; n != result; result += 8 )
      *(_QWORD *)result = -4096;
    a1[28] = 0;
    return result;
  }
  v30 = v25 - 1;
  if ( !v30 )
  {
    v31 = (_QWORD *)a1[27];
    LODWORD(v32) = 64;
LABEL_49:
    sub_C7D6A0((__int64)v31, 8 * v27, 8);
    v33 = ((((((((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v32 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v32 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v32 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v32 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 16;
    v34 = (v33
         | (((((((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v32 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v32 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v32 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v32 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1))
        + 1;
    *((_DWORD *)a1 + 58) = v34;
    result = sub_C7D670(8 * v34, 8);
    v35 = *((unsigned int *)a1 + 58);
    a1[28] = 0;
    a1[27] = result;
    for ( ii = result + 8 * v35; ii != result; result += 8 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
    return result;
  }
  _BitScanReverse(&v30, v30);
  v31 = (_QWORD *)a1[27];
  v32 = (unsigned int)(1 << (33 - (v30 ^ 0x1F)));
  if ( (int)v32 < 64 )
    v32 = 64;
  if ( (_DWORD)v32 != (_DWORD)v27 )
    goto LABEL_49;
  a1[28] = 0;
  result = (__int64)&v31[v32];
  do
  {
    if ( v31 )
      *v31 = -4096;
    ++v31;
  }
  while ( (_QWORD *)result != v31 );
  return result;
}
