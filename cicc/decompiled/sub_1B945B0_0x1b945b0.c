// Function: sub_1B945B0
// Address: 0x1b945b0
//
__int64 __fastcall sub_1B945B0(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  __int64 *v6; // r12
  __int64 v7; // r15
  __int64 *v8; // rbx
  __int64 *v9; // r14
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  void *v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // rdx
  unsigned __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r12
  unsigned __int64 v23; // rdi
  __int64 v24; // rax
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rdi
  __int64 v27; // rdi
  unsigned int v29; // ecx
  _QWORD *v30; // rdi
  unsigned int v31; // eax
  int v32; // eax
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rax
  int v35; // ebx
  __int64 v36; // r12
  _QWORD *v37; // rax
  __int64 v38; // rdx
  _QWORD *j; // rdx
  unsigned __int64 *v40; // rdx
  unsigned __int64 v41; // rcx
  unsigned __int64 *v42; // r12
  unsigned __int64 *v43; // rbx
  unsigned __int64 v44; // rdi
  unsigned __int64 *v45; // rbx
  unsigned __int64 v46; // rdi
  _QWORD *v47; // rax
  __int64 *v48; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v29 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v29 = 64;
  if ( v29 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 8);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v30 = *(_QWORD **)(a1 + 8);
  v31 = v2 - 1;
  if ( !v31 )
  {
    v36 = 2048;
    v35 = 128;
LABEL_51:
    j___libc_free_0(v30);
    *(_DWORD *)(a1 + 24) = v35;
    v37 = (_QWORD *)sub_22077B0(v36);
    v38 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v37;
    for ( j = &v37[2 * v38]; j != v37; v37 += 2 )
    {
      if ( v37 )
        *v37 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v31, v31);
  v32 = 1 << (33 - (v31 ^ 0x1F));
  if ( v32 < 64 )
    v32 = 64;
  if ( (_DWORD)v3 != v32 )
  {
    v33 = (4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1);
    v34 = ((v33 | (v33 >> 2)) >> 4) | v33 | (v33 >> 2) | ((((v33 | (v33 >> 2)) >> 4) | v33 | (v33 >> 2)) >> 8);
    v35 = (v34 | (v34 >> 16)) + 1;
    v36 = 16 * ((v34 | (v34 >> 16)) + 1);
    goto LABEL_51;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v47 = &v30[2 * (unsigned int)v3];
  do
  {
    if ( v30 )
      *v30 = -8;
    v30 += 2;
  }
  while ( v47 != v30 );
LABEL_7:
  v6 = *(__int64 **)(a1 + 32);
  v48 = *(__int64 **)(a1 + 40);
  if ( v6 != v48 )
  {
    do
    {
      v7 = *v6;
      v8 = *(__int64 **)(*v6 + 16);
      if ( *(__int64 **)(*v6 + 8) == v8 )
      {
        *(_BYTE *)(v7 + 160) = 1;
      }
      else
      {
        v9 = *(__int64 **)(*v6 + 8);
        do
        {
          v10 = *v9++;
          sub_1B93FC0(v10);
        }
        while ( v8 != v9 );
        *(_BYTE *)(v7 + 160) = 1;
        v11 = *(_QWORD *)(v7 + 8);
        if ( *(_QWORD *)(v7 + 16) != v11 )
          *(_QWORD *)(v7 + 16) = v11;
      }
      v12 = *(_QWORD *)(v7 + 32);
      if ( v12 != *(_QWORD *)(v7 + 40) )
        *(_QWORD *)(v7 + 40) = v12;
      ++*(_QWORD *)(v7 + 56);
      v13 = *(void **)(v7 + 72);
      if ( v13 == *(void **)(v7 + 64) )
      {
        *(_QWORD *)v7 = 0;
      }
      else
      {
        v14 = 4 * (*(_DWORD *)(v7 + 84) - *(_DWORD *)(v7 + 88));
        v15 = *(unsigned int *)(v7 + 80);
        if ( v14 < 0x20 )
          v14 = 32;
        if ( (unsigned int)v15 > v14 )
          sub_16CC920(v7 + 56);
        else
          memset(v13, -1, 8 * v15);
        v16 = *(_QWORD *)(v7 + 72);
        v17 = *(_QWORD *)(v7 + 64);
        *(_QWORD *)v7 = 0;
        if ( v16 != v17 )
          _libc_free(v16);
      }
      v18 = *(_QWORD *)(v7 + 32);
      if ( v18 )
        j_j___libc_free_0(v18, *(_QWORD *)(v7 + 48) - v18);
      v19 = *(_QWORD *)(v7 + 8);
      if ( v19 )
        j_j___libc_free_0(v19, *(_QWORD *)(v7 + 24) - v19);
      ++v6;
    }
    while ( v48 != v6 );
    v20 = *(_QWORD *)(a1 + 32);
    if ( *(_QWORD *)(a1 + 40) != v20 )
      *(_QWORD *)(a1 + 40) = v20;
  }
  v21 = *(unsigned __int64 **)(a1 + 120);
  v22 = &v21[2 * *(unsigned int *)(a1 + 128)];
  while ( v22 != v21 )
  {
    v23 = *v21;
    v21 += 2;
    _libc_free(v23);
  }
  *(_DWORD *)(a1 + 128) = 0;
  v24 = *(unsigned int *)(a1 + 80);
  if ( !(_DWORD)v24 )
    goto LABEL_32;
  *(_QWORD *)(a1 + 136) = 0;
  v40 = *(unsigned __int64 **)(a1 + 72);
  v41 = *v40;
  v42 = &v40[v24];
  v43 = v40 + 1;
  *(_QWORD *)(a1 + 56) = *v40;
  *(_QWORD *)(a1 + 64) = v41 + 4096;
  if ( v42 != v40 + 1 )
  {
    do
    {
      v44 = *v43++;
      _libc_free(v44);
    }
    while ( v42 != v43 );
    v40 = *(unsigned __int64 **)(a1 + 72);
  }
  *(_DWORD *)(a1 + 80) = 1;
  _libc_free(*v40);
  v45 = *(unsigned __int64 **)(a1 + 120);
  v25 = (unsigned __int64)&v45[2 * *(unsigned int *)(a1 + 128)];
  if ( v45 != (unsigned __int64 *)v25 )
  {
    do
    {
      v46 = *v45;
      v45 += 2;
      _libc_free(v46);
    }
    while ( v45 != (unsigned __int64 *)v25 );
LABEL_32:
    v25 = *(_QWORD *)(a1 + 120);
  }
  if ( v25 != a1 + 136 )
    _libc_free(v25);
  v26 = *(_QWORD *)(a1 + 72);
  if ( v26 != a1 + 88 )
    _libc_free(v26);
  v27 = *(_QWORD *)(a1 + 32);
  if ( v27 )
    j_j___libc_free_0(v27, *(_QWORD *)(a1 + 48) - v27);
  return j___libc_free_0(*(_QWORD *)(a1 + 8));
}
