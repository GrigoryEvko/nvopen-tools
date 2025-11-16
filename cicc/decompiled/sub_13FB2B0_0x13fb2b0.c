// Function: sub_13FB2B0
// Address: 0x13fb2b0
//
void __fastcall sub_13FB2B0(__int64 a1)
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
  unsigned int v25; // ecx
  _QWORD *v26; // rdi
  unsigned int v27; // eax
  int v28; // eax
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  int v31; // ebx
  __int64 v32; // r12
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _QWORD *j; // rdx
  _QWORD *v36; // rbx
  __int64 v37; // rdx
  unsigned __int64 *v38; // r12
  unsigned __int64 *v39; // rbx
  unsigned __int64 v40; // rdi
  _QWORD *v41; // rax
  __int64 *v42; // [rsp+8h] [rbp-38h]

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
  v25 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v25 = 64;
  if ( (unsigned int)v3 <= v25 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 8);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v26 = *(_QWORD **)(a1 + 8);
  v27 = v2 - 1;
  if ( !v27 )
  {
    v32 = 2048;
    v31 = 128;
LABEL_44:
    j___libc_free_0(v26);
    *(_DWORD *)(a1 + 24) = v31;
    v33 = (_QWORD *)sub_22077B0(v32);
    v34 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v33;
    for ( j = &v33[2 * v34]; j != v33; v33 += 2 )
    {
      if ( v33 )
        *v33 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v27, v27);
  v28 = 1 << (33 - (v27 ^ 0x1F));
  if ( v28 < 64 )
    v28 = 64;
  if ( (_DWORD)v3 != v28 )
  {
    v29 = (4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1);
    v30 = ((v29 | (v29 >> 2)) >> 4) | v29 | (v29 >> 2) | ((((v29 | (v29 >> 2)) >> 4) | v29 | (v29 >> 2)) >> 8);
    v31 = (v30 | (v30 >> 16)) + 1;
    v32 = 16 * ((v30 | (v30 >> 16)) + 1);
    goto LABEL_44;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v41 = &v26[2 * (unsigned int)v3];
  do
  {
    if ( v26 )
      *v26 = -8;
    v26 += 2;
  }
  while ( v41 != v26 );
LABEL_7:
  v6 = *(__int64 **)(a1 + 32);
  v42 = *(__int64 **)(a1 + 40);
  if ( v6 != v42 )
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
          sub_13FACC0(v10);
        }
        while ( v8 != v9 );
        *(_BYTE *)(v7 + 160) = 1;
        v11 = *(_QWORD *)(v7 + 8);
        if ( v11 != *(_QWORD *)(v7 + 16) )
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
        if ( v17 != v16 )
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
    while ( v42 != v6 );
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
  if ( (_DWORD)v24 )
  {
    *(_QWORD *)(a1 + 136) = 0;
    v36 = *(_QWORD **)(a1 + 72);
    v37 = *v36;
    v38 = &v36[v24];
    v39 = v36 + 1;
    *(_QWORD *)(a1 + 56) = v37;
    *(_QWORD *)(a1 + 64) = v37 + 4096;
    while ( v38 != v39 )
    {
      v40 = *v39++;
      _libc_free(v40);
    }
    *(_DWORD *)(a1 + 80) = 1;
  }
}
