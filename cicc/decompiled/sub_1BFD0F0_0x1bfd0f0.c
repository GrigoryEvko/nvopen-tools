// Function: sub_1BFD0F0
// Address: 0x1bfd0f0
//
__int64 __fastcall sub_1BFD0F0(__int64 a1)
{
  int v2; // ecx
  __int64 v3; // rax
  _QWORD *v4; // rdi
  _QWORD *i; // rax
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *k; // rdx
  int v10; // eax
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 n; // rdx
  unsigned int v14; // ecx
  _DWORD *v15; // rdi
  unsigned int v16; // eax
  int v17; // eax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  int v20; // ebx
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 ii; // rdx
  unsigned int v24; // ecx
  _QWORD *v25; // rdi
  unsigned int v26; // eax
  int v27; // eax
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rax
  int v30; // ebx
  __int64 v31; // r12
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *m; // rdx
  _QWORD *v35; // r12
  _QWORD *v36; // rax
  _QWORD *v37; // rbx
  unsigned __int64 **v38; // r13
  unsigned __int64 *v39; // r15
  unsigned __int64 *v40; // r15
  unsigned __int64 *v41; // r15
  unsigned __int64 *v42; // r15
  unsigned int v43; // edx
  _QWORD *v44; // r8
  unsigned int v45; // ecx
  unsigned int v46; // edx
  int v47; // edx
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rax
  int v50; // ebx
  __int64 v51; // r12
  _QWORD *v52; // rax
  __int64 v53; // rdx
  _QWORD *j; // rdx
  _QWORD *v55; // rax
  _QWORD *v56; // rax

  v2 = *(_DWORD *)(a1 + 96);
  if ( !v2 )
  {
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_3;
  }
  v4 = *(_QWORD **)(a1 + 88);
  v35 = &v4[2 * *(unsigned int *)(a1 + 104)];
  if ( v4 == v35 )
    goto LABEL_52;
  v36 = v4;
  while ( 1 )
  {
    v37 = v36;
    if ( *v36 != -8 && *v36 != -16 )
      break;
    v36 += 2;
    if ( v35 == v36 )
      goto LABEL_52;
  }
  if ( v35 == v36 )
  {
LABEL_52:
    ++*(_QWORD *)(a1 + 80);
  }
  else
  {
    do
    {
      v38 = (unsigned __int64 **)v37[1];
      if ( v38 )
      {
        v39 = *v38;
        if ( *v38 )
        {
          _libc_free(*v39);
          j_j___libc_free_0(v39, 24);
        }
        v40 = v38[1];
        if ( v40 )
        {
          _libc_free(*v40);
          j_j___libc_free_0(v40, 24);
        }
        v41 = v38[2];
        if ( v41 )
        {
          _libc_free(*v41);
          j_j___libc_free_0(v41, 24);
        }
        v42 = v38[3];
        if ( v42 )
        {
          _libc_free(*v42);
          j_j___libc_free_0(v42, 24);
        }
        j_j___libc_free_0(v38, 32);
      }
      v37 += 2;
      if ( v37 == v35 )
        break;
      while ( *v37 == -8 || *v37 == -16 )
      {
        v37 += 2;
        if ( v35 == v37 )
          goto LABEL_68;
      }
    }
    while ( v35 != v37 );
LABEL_68:
    v2 = *(_DWORD *)(a1 + 96);
    ++*(_QWORD *)(a1 + 80);
    if ( !v2 )
    {
LABEL_3:
      if ( !*(_DWORD *)(a1 + 100) )
        goto LABEL_8;
      v3 = *(unsigned int *)(a1 + 104);
      v4 = *(_QWORD **)(a1 + 88);
      if ( (unsigned int)v3 > 0x40 )
      {
        j___libc_free_0(v4);
        *(_QWORD *)(a1 + 88) = 0;
        *(_QWORD *)(a1 + 96) = 0;
        *(_DWORD *)(a1 + 104) = 0;
        goto LABEL_8;
      }
      goto LABEL_5;
    }
    v4 = *(_QWORD **)(a1 + 88);
  }
  v43 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 104);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v43 = 64;
  if ( (unsigned int)v3 <= v43 )
  {
LABEL_5:
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 96) = 0;
    goto LABEL_8;
  }
  v44 = v4;
  v45 = v2 - 1;
  if ( !v45 )
  {
    v51 = 2048;
    v50 = 128;
LABEL_78:
    j___libc_free_0(v4);
    *(_DWORD *)(a1 + 104) = v50;
    v52 = (_QWORD *)sub_22077B0(v51);
    v53 = *(unsigned int *)(a1 + 104);
    *(_QWORD *)(a1 + 96) = 0;
    *(_QWORD *)(a1 + 88) = v52;
    for ( j = &v52[2 * v53]; j != v52; v52 += 2 )
    {
      if ( v52 )
        *v52 = -8;
    }
    goto LABEL_8;
  }
  _BitScanReverse(&v46, v45);
  v47 = 1 << (33 - (v46 ^ 0x1F));
  if ( v47 < 64 )
    v47 = 64;
  if ( (_DWORD)v3 != v47 )
  {
    v48 = (4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1);
    v49 = ((v48 | (v48 >> 2)) >> 4) | v48 | (v48 >> 2) | ((((v48 | (v48 >> 2)) >> 4) | v48 | (v48 >> 2)) >> 8);
    v50 = (v49 | (v49 >> 16)) + 1;
    v51 = 16 * ((v49 | (v49 >> 16)) + 1);
    goto LABEL_78;
  }
  *(_QWORD *)(a1 + 96) = 0;
  v56 = &v4[2 * v3];
  do
  {
    if ( v44 )
      *v44 = -8;
    v44 += 2;
  }
  while ( v56 != v44 );
LABEL_8:
  v6 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_14;
    v7 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v7 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 16));
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      goto LABEL_14;
    }
    goto LABEL_11;
  }
  v24 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v24 = 64;
  if ( v24 >= (unsigned int)v7 )
  {
LABEL_11:
    v8 = *(_QWORD **)(a1 + 16);
    for ( k = &v8[2 * v7]; k != v8; v8 += 2 )
      *v8 = -8;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_14;
  }
  v25 = *(_QWORD **)(a1 + 16);
  v26 = v6 - 1;
  if ( !v26 )
  {
    v31 = 2048;
    v30 = 128;
LABEL_42:
    j___libc_free_0(v25);
    *(_DWORD *)(a1 + 32) = v30;
    v32 = (_QWORD *)sub_22077B0(v31);
    v33 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = v32;
    for ( m = &v32[2 * v33]; m != v32; v32 += 2 )
    {
      if ( v32 )
        *v32 = -8;
    }
    goto LABEL_14;
  }
  _BitScanReverse(&v26, v26);
  v27 = 1 << (33 - (v26 ^ 0x1F));
  if ( v27 < 64 )
    v27 = 64;
  if ( (_DWORD)v7 != v27 )
  {
    v28 = (4 * v27 / 3u + 1) | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1);
    v29 = ((v28 | (v28 >> 2)) >> 4) | v28 | (v28 >> 2) | ((((v28 | (v28 >> 2)) >> 4) | v28 | (v28 >> 2)) >> 8);
    v30 = (v29 | (v29 >> 16)) + 1;
    v31 = 16 * ((v29 | (v29 >> 16)) + 1);
    goto LABEL_42;
  }
  *(_QWORD *)(a1 + 24) = 0;
  v55 = &v25[2 * (unsigned int)v7];
  do
  {
    if ( v25 )
      *v25 = -8;
    v25 += 2;
  }
  while ( v55 != v25 );
LABEL_14:
  v10 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( !v10 )
  {
    result = *(unsigned int *)(a1 + 60);
    if ( !(_DWORD)result )
      goto LABEL_20;
    v12 = *(unsigned int *)(a1 + 64);
    if ( (unsigned int)v12 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 48));
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = 0;
      *(_DWORD *)(a1 + 64) = 0;
      goto LABEL_20;
    }
    goto LABEL_17;
  }
  v14 = 4 * v10;
  v12 = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)(4 * v10) < 0x40 )
    v14 = 64;
  if ( (unsigned int)v12 <= v14 )
  {
LABEL_17:
    result = *(_QWORD *)(a1 + 48);
    for ( n = result + 16 * v12; n != result; result += 16 )
      *(_DWORD *)result = 0x7FFFFFFF;
    *(_QWORD *)(a1 + 56) = 0;
    goto LABEL_20;
  }
  v15 = *(_DWORD **)(a1 + 48);
  v16 = v10 - 1;
  if ( !v16 )
  {
    v21 = 2048;
    v20 = 128;
LABEL_29:
    j___libc_free_0(v15);
    *(_DWORD *)(a1 + 64) = v20;
    result = sub_22077B0(v21);
    v22 = *(unsigned int *)(a1 + 64);
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 48) = result;
    for ( ii = result + 16 * v22; ii != result; result += 16 )
    {
      if ( result )
        *(_DWORD *)result = 0x7FFFFFFF;
    }
    goto LABEL_20;
  }
  _BitScanReverse(&v16, v16);
  v17 = 1 << (33 - (v16 ^ 0x1F));
  if ( v17 < 64 )
    v17 = 64;
  if ( (_DWORD)v12 != v17 )
  {
    v18 = (4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1);
    v19 = ((v18 | (v18 >> 2)) >> 4) | v18 | (v18 >> 2) | ((((v18 | (v18 >> 2)) >> 4) | v18 | (v18 >> 2)) >> 8);
    v20 = (v19 | (v19 >> 16)) + 1;
    v21 = 16 * ((v19 | (v19 >> 16)) + 1);
    goto LABEL_29;
  }
  *(_QWORD *)(a1 + 56) = 0;
  result = (__int64)&v15[4 * (unsigned int)v12];
  do
  {
    if ( v15 )
      *v15 = 0x7FFFFFFF;
    v15 += 4;
  }
  while ( (_DWORD *)result != v15 );
LABEL_20:
  *(_DWORD *)a1 = 0;
  *(_BYTE *)(a1 + 72) = 0;
  return result;
}
