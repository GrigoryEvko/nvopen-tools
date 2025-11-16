// Function: sub_39256D0
// Address: 0x39256d0
//
void __fastcall sub_39256D0(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 *v3; // r15
  unsigned __int64 *v4; // r14
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 *v9; // r15
  unsigned __int64 *v10; // r14
  unsigned __int64 *v11; // r13
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  int v15; // eax
  __int64 v16; // rdx
  _QWORD *v17; // rax
  _QWORD *i; // rdx
  int v19; // eax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  _QWORD *k; // rdx
  unsigned int v23; // ecx
  _QWORD *v24; // rdi
  unsigned int v25; // eax
  int v26; // eax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  int v29; // r13d
  unsigned __int64 v30; // r12
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _QWORD *m; // rdx
  unsigned int v34; // ecx
  _QWORD *v35; // rdi
  unsigned int v36; // eax
  int v37; // eax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rax
  int v40; // r13d
  unsigned __int64 v41; // r12
  _QWORD *v42; // rax
  __int64 v43; // rdx
  _QWORD *j; // rdx
  _QWORD *v45; // rax
  _QWORD *v46; // rax

  v2 = *(_QWORD *)(a1 + 24);
  *(_OWORD *)(a1 + 32) = 0;
  v3 = *(unsigned __int64 **)(a1 + 56);
  *(_QWORD *)(a1 + 48) = 0;
  v4 = *(unsigned __int64 **)(a1 + 64);
  *(_WORD *)(a1 + 32) = *(_DWORD *)(v2 + 8);
  if ( v3 != v4 )
  {
    v5 = v3;
    do
    {
      v6 = *v5;
      if ( *v5 )
      {
        v7 = *(_QWORD *)(v6 + 96);
        if ( v7 )
          j_j___libc_free_0(v7);
        v8 = *(_QWORD *)(v6 + 40);
        if ( v8 != v6 + 56 )
          j_j___libc_free_0(v8);
        j_j___libc_free_0(v6);
      }
      ++v5;
    }
    while ( v4 != v5 );
    *(_QWORD *)(a1 + 64) = v3;
  }
  v9 = *(unsigned __int64 **)(a1 + 80);
  v10 = *(unsigned __int64 **)(a1 + 88);
  if ( v9 != v10 )
  {
    v11 = *(unsigned __int64 **)(a1 + 80);
    do
    {
      v12 = *v11;
      if ( *v11 )
      {
        v13 = *(_QWORD *)(v12 + 56);
        if ( v13 != v12 + 72 )
          _libc_free(v13);
        v14 = *(_QWORD *)(v12 + 24);
        if ( v14 != v12 + 40 )
          _libc_free(v14);
        j_j___libc_free_0(v12);
      }
      ++v11;
    }
    while ( v10 != v11 );
    *(_QWORD *)(a1 + 88) = v9;
  }
  sub_167FC70(a1 + 104);
  v15 = *(_DWORD *)(a1 + 176);
  ++*(_QWORD *)(a1 + 160);
  if ( !v15 )
  {
    if ( !*(_DWORD *)(a1 + 180) )
      goto LABEL_27;
    v16 = *(unsigned int *)(a1 + 184);
    if ( (unsigned int)v16 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 168));
      *(_QWORD *)(a1 + 168) = 0;
      *(_QWORD *)(a1 + 176) = 0;
      *(_DWORD *)(a1 + 184) = 0;
      goto LABEL_27;
    }
    goto LABEL_24;
  }
  v34 = 4 * v15;
  v16 = *(unsigned int *)(a1 + 184);
  if ( (unsigned int)(4 * v15) < 0x40 )
    v34 = 64;
  if ( (unsigned int)v16 <= v34 )
  {
LABEL_24:
    v17 = *(_QWORD **)(a1 + 168);
    for ( i = &v17[2 * v16]; i != v17; v17 += 2 )
      *v17 = -8;
    *(_QWORD *)(a1 + 176) = 0;
    goto LABEL_27;
  }
  v35 = *(_QWORD **)(a1 + 168);
  v36 = v15 - 1;
  if ( !v36 )
  {
    v41 = 2048;
    v40 = 128;
LABEL_55:
    j___libc_free_0((unsigned __int64)v35);
    *(_DWORD *)(a1 + 184) = v40;
    v42 = (_QWORD *)sub_22077B0(v41);
    v43 = *(unsigned int *)(a1 + 184);
    *(_QWORD *)(a1 + 176) = 0;
    *(_QWORD *)(a1 + 168) = v42;
    for ( j = &v42[2 * v43]; j != v42; v42 += 2 )
    {
      if ( v42 )
        *v42 = -8;
    }
    goto LABEL_27;
  }
  _BitScanReverse(&v36, v36);
  v37 = 1 << (33 - (v36 ^ 0x1F));
  if ( v37 < 64 )
    v37 = 64;
  if ( (_DWORD)v16 != v37 )
  {
    v38 = (((4 * v37 / 3u + 1) | ((unsigned __int64)(4 * v37 / 3u + 1) >> 1)) >> 2)
        | (4 * v37 / 3u + 1)
        | ((unsigned __int64)(4 * v37 / 3u + 1) >> 1)
        | (((((4 * v37 / 3u + 1) | ((unsigned __int64)(4 * v37 / 3u + 1) >> 1)) >> 2)
          | (4 * v37 / 3u + 1)
          | ((unsigned __int64)(4 * v37 / 3u + 1) >> 1)) >> 4);
    v39 = (v38 >> 8) | v38;
    v40 = (v39 | (v39 >> 16)) + 1;
    v41 = 16 * ((v39 | (v39 >> 16)) + 1);
    goto LABEL_55;
  }
  *(_QWORD *)(a1 + 176) = 0;
  v45 = &v35[2 * (unsigned int)v16];
  do
  {
    if ( v35 )
      *v35 = -8;
    v35 += 2;
  }
  while ( v45 != v35 );
LABEL_27:
  v19 = *(_DWORD *)(a1 + 208);
  ++*(_QWORD *)(a1 + 192);
  if ( !v19 )
  {
    if ( !*(_DWORD *)(a1 + 212) )
      return;
    v20 = *(unsigned int *)(a1 + 216);
    if ( (unsigned int)v20 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 200));
      *(_QWORD *)(a1 + 200) = 0;
      *(_QWORD *)(a1 + 208) = 0;
      *(_DWORD *)(a1 + 216) = 0;
      return;
    }
    goto LABEL_30;
  }
  v23 = 4 * v19;
  v20 = *(unsigned int *)(a1 + 216);
  if ( (unsigned int)(4 * v19) < 0x40 )
    v23 = 64;
  if ( v23 >= (unsigned int)v20 )
  {
LABEL_30:
    v21 = *(_QWORD **)(a1 + 200);
    for ( k = &v21[2 * v20]; k != v21; v21 += 2 )
      *v21 = -8;
    *(_QWORD *)(a1 + 208) = 0;
    return;
  }
  v24 = *(_QWORD **)(a1 + 200);
  v25 = v19 - 1;
  if ( !v25 )
  {
    v30 = 2048;
    v29 = 128;
LABEL_42:
    j___libc_free_0((unsigned __int64)v24);
    *(_DWORD *)(a1 + 216) = v29;
    v31 = (_QWORD *)sub_22077B0(v30);
    v32 = *(unsigned int *)(a1 + 216);
    *(_QWORD *)(a1 + 208) = 0;
    *(_QWORD *)(a1 + 200) = v31;
    for ( m = &v31[2 * v32]; m != v31; v31 += 2 )
    {
      if ( v31 )
        *v31 = -8;
    }
    return;
  }
  _BitScanReverse(&v25, v25);
  v26 = 1 << (33 - (v25 ^ 0x1F));
  if ( v26 < 64 )
    v26 = 64;
  if ( (_DWORD)v20 != v26 )
  {
    v27 = (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
        | (4 * v26 / 3u + 1)
        | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)
        | (((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
          | (4 * v26 / 3u + 1)
          | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4);
    v28 = (v27 >> 8) | v27;
    v29 = (v28 | (v28 >> 16)) + 1;
    v30 = 16 * ((v28 | (v28 >> 16)) + 1);
    goto LABEL_42;
  }
  *(_QWORD *)(a1 + 208) = 0;
  v46 = &v24[2 * (unsigned int)v20];
  do
  {
    if ( v24 )
      *v24 = -8;
    v24 += 2;
  }
  while ( v46 != v24 );
}
