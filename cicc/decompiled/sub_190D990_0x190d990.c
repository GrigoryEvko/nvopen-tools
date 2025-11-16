// Function: sub_190D990
// Address: 0x190d990
//
__int64 __fastcall sub_190D990(__int64 a1, __int64 a2, __int64 i, __int64 a4, int a5, int a6)
{
  int v6; // eax
  __int64 v8; // rcx
  _QWORD *v9; // rax
  int v10; // eax
  __int64 v11; // rdx
  _DWORD *v12; // rax
  _DWORD *j; // rdx
  int v14; // eax
  __int64 v15; // rdx
  _DWORD *v16; // rax
  _DWORD *m; // rdx
  __int64 v18; // r13
  __int64 v19; // r14
  __int64 v20; // r12
  unsigned __int64 v21; // rdi
  __int64 result; // rax
  unsigned int v23; // ecx
  __int64 v24; // rdi
  unsigned int v25; // eax
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  int v29; // r13d
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 n; // rdx
  unsigned int v34; // ecx
  _DWORD *v35; // rdi
  unsigned int v36; // eax
  int v37; // eax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rax
  int v40; // r13d
  __int64 v41; // r12
  _DWORD *v42; // rax
  __int64 v43; // rdx
  _DWORD *k; // rdx
  _QWORD *v45; // rdi
  unsigned int v46; // eax
  int v47; // eax
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rax
  int v50; // r13d
  __int64 v51; // r12
  _QWORD *v52; // rax
  __int64 v53; // rdx
  _DWORD *v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // rax

  v6 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v6 )
  {
    v8 = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)v8 )
      goto LABEL_7;
    i = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)i > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v8 = (unsigned int)(4 * v6);
  a2 = 64;
  i = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)v8 < 0x40 )
    v8 = 64;
  if ( (unsigned int)i <= (unsigned int)v8 )
  {
LABEL_4:
    v9 = *(_QWORD **)(a1 + 8);
    for ( i = (__int64)&v9[2 * i]; (_QWORD *)i != v9; v9 += 2 )
      *v9 = -8;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v45 = *(_QWORD **)(a1 + 8);
  v46 = v6 - 1;
  if ( !v46 )
  {
    v51 = 2048;
    v50 = 128;
LABEL_62:
    j___libc_free_0(v45);
    *(_DWORD *)(a1 + 24) = v50;
    v52 = (_QWORD *)sub_22077B0(v51);
    v53 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v52;
    for ( i = (__int64)&v52[2 * v53]; (_QWORD *)i != v52; v52 += 2 )
    {
      if ( v52 )
        *v52 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v46, v46);
  v8 = 33 - (v46 ^ 0x1F);
  v47 = 1 << (33 - (v46 ^ 0x1F));
  if ( v47 < 64 )
    v47 = 64;
  if ( (_DWORD)i != v47 )
  {
    v48 = (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
        | (4 * v47 / 3u + 1)
        | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)
        | (((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
          | (4 * v47 / 3u + 1)
          | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4);
    v49 = (v48 >> 8) | v48;
    v50 = (v49 | (v49 >> 16)) + 1;
    v51 = 16 * ((v49 | (v49 >> 16)) + 1);
    goto LABEL_62;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v55 = &v45[2 * (unsigned int)i];
  do
  {
    if ( v45 )
      *v45 = -8;
    v45 += 2;
  }
  while ( v55 != v45 );
LABEL_7:
  sub_190D6F0(a1 + 32, a2, i, v8, a5, a6);
  v10 = *(_DWORD *)(a1 + 136);
  ++*(_QWORD *)(a1 + 120);
  if ( !v10 )
  {
    if ( !*(_DWORD *)(a1 + 140) )
      goto LABEL_13;
    v11 = *(unsigned int *)(a1 + 144);
    if ( (unsigned int)v11 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 128));
      *(_QWORD *)(a1 + 128) = 0;
      *(_QWORD *)(a1 + 136) = 0;
      *(_DWORD *)(a1 + 144) = 0;
      goto LABEL_13;
    }
    goto LABEL_10;
  }
  v34 = 4 * v10;
  v11 = *(unsigned int *)(a1 + 144);
  if ( (unsigned int)(4 * v10) < 0x40 )
    v34 = 64;
  if ( (unsigned int)v11 <= v34 )
  {
LABEL_10:
    v12 = *(_DWORD **)(a1 + 128);
    for ( j = &v12[4 * v11]; j != v12; v12 += 4 )
      *v12 = -1;
    *(_QWORD *)(a1 + 136) = 0;
    goto LABEL_13;
  }
  v35 = *(_DWORD **)(a1 + 128);
  v36 = v10 - 1;
  if ( !v36 )
  {
    v41 = 2048;
    v40 = 128;
LABEL_49:
    j___libc_free_0(v35);
    *(_DWORD *)(a1 + 144) = v40;
    v42 = (_DWORD *)sub_22077B0(v41);
    v43 = *(unsigned int *)(a1 + 144);
    *(_QWORD *)(a1 + 136) = 0;
    *(_QWORD *)(a1 + 128) = v42;
    for ( k = &v42[4 * v43]; k != v42; v42 += 4 )
    {
      if ( v42 )
        *v42 = -1;
    }
    goto LABEL_13;
  }
  _BitScanReverse(&v36, v36);
  v37 = 1 << (33 - (v36 ^ 0x1F));
  if ( v37 < 64 )
    v37 = 64;
  if ( (_DWORD)v11 != v37 )
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
    goto LABEL_49;
  }
  *(_QWORD *)(a1 + 136) = 0;
  v54 = &v35[4 * (unsigned int)v11];
  do
  {
    if ( v35 )
      *v35 = -1;
    v35 += 4;
  }
  while ( v54 != v35 );
LABEL_13:
  v14 = *(_DWORD *)(a1 + 168);
  ++*(_QWORD *)(a1 + 152);
  if ( !v14 )
  {
    if ( !*(_DWORD *)(a1 + 172) )
      goto LABEL_19;
    v15 = *(unsigned int *)(a1 + 176);
    if ( (unsigned int)v15 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 160));
      *(_QWORD *)(a1 + 160) = 0;
      *(_QWORD *)(a1 + 168) = 0;
      *(_DWORD *)(a1 + 176) = 0;
      goto LABEL_19;
    }
    goto LABEL_16;
  }
  v23 = 4 * v14;
  v15 = *(unsigned int *)(a1 + 176);
  if ( (unsigned int)(4 * v14) < 0x40 )
    v23 = 64;
  if ( (unsigned int)v15 <= v23 )
  {
LABEL_16:
    v16 = *(_DWORD **)(a1 + 160);
    for ( m = &v16[6 * v15]; m != v16; *((_QWORD *)v16 - 2) = -8 )
    {
      *v16 = -1;
      v16 += 6;
    }
    *(_QWORD *)(a1 + 168) = 0;
    goto LABEL_19;
  }
  v24 = *(_QWORD *)(a1 + 160);
  v25 = v14 - 1;
  if ( !v25 )
  {
    v30 = 3072;
    v29 = 128;
LABEL_36:
    j___libc_free_0(v24);
    *(_DWORD *)(a1 + 176) = v29;
    v31 = sub_22077B0(v30);
    v32 = *(unsigned int *)(a1 + 176);
    *(_QWORD *)(a1 + 168) = 0;
    *(_QWORD *)(a1 + 160) = v31;
    for ( n = v31 + 24 * v32; n != v31; v31 += 24 )
    {
      if ( v31 )
      {
        *(_DWORD *)v31 = -1;
        *(_QWORD *)(v31 + 8) = -8;
      }
    }
    goto LABEL_19;
  }
  _BitScanReverse(&v25, v25);
  v26 = (unsigned int)(1 << (33 - (v25 ^ 0x1F)));
  if ( (int)v26 < 64 )
    v26 = 64;
  if ( (_DWORD)v26 != (_DWORD)v15 )
  {
    v27 = (4 * (int)v26 / 3u + 1) | ((unsigned __int64)(4 * (int)v26 / 3u + 1) >> 1);
    v28 = ((((v27 >> 2) | v27 | (((v27 >> 2) | v27) >> 4)) >> 8)
         | (v27 >> 2)
         | v27
         | (((v27 >> 2) | v27) >> 4)
         | (((((v27 >> 2) | v27 | (((v27 >> 2) | v27) >> 4)) >> 8) | (v27 >> 2) | v27 | (((v27 >> 2) | v27) >> 4)) >> 16))
        + 1;
    v29 = v28;
    v30 = 24 * v28;
    goto LABEL_36;
  }
  *(_QWORD *)(a1 + 168) = 0;
  v56 = v24 + 24 * v26;
  do
  {
    if ( v24 )
    {
      *(_DWORD *)v24 = -1;
      *(_QWORD *)(v24 + 8) = -8;
    }
    v24 += 24;
  }
  while ( v56 != v24 );
LABEL_19:
  v18 = *(_QWORD *)(a1 + 72);
  v19 = *(_QWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 208) = 1;
  if ( v18 != v19 )
  {
    v20 = v18;
    do
    {
      v21 = *(_QWORD *)(v20 + 24);
      if ( v21 != v20 + 40 )
        _libc_free(v21);
      v20 += 56;
    }
    while ( v19 != v20 );
    *(_QWORD *)(a1 + 80) = v18;
  }
  result = *(_QWORD *)(a1 + 96);
  if ( result != *(_QWORD *)(a1 + 104) )
    *(_QWORD *)(a1 + 104) = result;
  *(_DWORD *)(a1 + 64) = 0;
  return result;
}
