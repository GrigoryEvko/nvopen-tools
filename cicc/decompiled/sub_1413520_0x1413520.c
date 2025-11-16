// Function: sub_1413520
// Address: 0x1413520
//
void __fastcall sub_1413520(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *k; // rdx
  unsigned __int64 *v10; // rbx
  unsigned __int64 *v11; // r13
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  unsigned int v14; // ecx
  _QWORD *v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  int v20; // ebx
  __int64 v21; // r13
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _QWORD *m; // rdx
  unsigned int v25; // ecx
  _QWORD *v26; // rdi
  unsigned int v27; // eax
  int v28; // eax
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  int v31; // ebx
  __int64 v32; // r13
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _QWORD *j; // rdx
  _QWORD *v36; // rbx
  __int64 v37; // rdx
  unsigned __int64 *v38; // r13
  unsigned __int64 *v39; // rbx
  unsigned __int64 v40; // rdi
  _QWORD *v41; // rax
  _QWORD *v42; // rax

  v2 = *(_DWORD *)(a1 + 312);
  ++*(_QWORD *)(a1 + 296);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 316) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 320);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 304));
      *(_QWORD *)(a1 + 304) = 0;
      *(_QWORD *)(a1 + 312) = 0;
      *(_DWORD *)(a1 + 320) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v25 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 320);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v25 = 64;
  if ( (unsigned int)v3 <= v25 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 304);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 312) = 0;
    goto LABEL_7;
  }
  v26 = *(_QWORD **)(a1 + 304);
  v27 = v2 - 1;
  if ( !v27 )
  {
    v32 = 2048;
    v31 = 128;
LABEL_38:
    j___libc_free_0(v26);
    *(_DWORD *)(a1 + 320) = v31;
    v33 = (_QWORD *)sub_22077B0(v32);
    v34 = *(unsigned int *)(a1 + 320);
    *(_QWORD *)(a1 + 312) = 0;
    *(_QWORD *)(a1 + 304) = v33;
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
    v29 = (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
        | (4 * v28 / 3u + 1)
        | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)
        | (((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
          | (4 * v28 / 3u + 1)
          | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4);
    v30 = (v29 >> 8) | v29;
    v31 = (v30 | (v30 >> 16)) + 1;
    v32 = 16 * ((v30 | (v30 >> 16)) + 1);
    goto LABEL_38;
  }
  *(_QWORD *)(a1 + 312) = 0;
  v41 = &v26[2 * (unsigned int)v3];
  do
  {
    if ( v26 )
      *v26 = -8;
    v26 += 2;
  }
  while ( v41 != v26 );
LABEL_7:
  v6 = *(_DWORD *)(a1 + 344);
  ++*(_QWORD *)(a1 + 328);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 348) )
      goto LABEL_13;
    v7 = *(unsigned int *)(a1 + 352);
    if ( (unsigned int)v7 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 336));
      *(_QWORD *)(a1 + 336) = 0;
      *(_QWORD *)(a1 + 344) = 0;
      *(_DWORD *)(a1 + 352) = 0;
      goto LABEL_13;
    }
    goto LABEL_10;
  }
  v14 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 352);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v14 = 64;
  if ( v14 >= (unsigned int)v7 )
  {
LABEL_10:
    v8 = *(_QWORD **)(a1 + 336);
    for ( k = &v8[2 * v7]; k != v8; v8 += 2 )
      *v8 = -8;
    *(_QWORD *)(a1 + 344) = 0;
    goto LABEL_13;
  }
  v15 = *(_QWORD **)(a1 + 336);
  v16 = v6 - 1;
  if ( !v16 )
  {
    v21 = 2048;
    v20 = 128;
LABEL_25:
    j___libc_free_0(v15);
    *(_DWORD *)(a1 + 352) = v20;
    v22 = (_QWORD *)sub_22077B0(v21);
    v23 = *(unsigned int *)(a1 + 352);
    *(_QWORD *)(a1 + 344) = 0;
    *(_QWORD *)(a1 + 336) = v22;
    for ( m = &v22[2 * v23]; m != v22; v22 += 2 )
    {
      if ( v22 )
        *v22 = -8;
    }
    goto LABEL_13;
  }
  _BitScanReverse(&v16, v16);
  v17 = (unsigned int)(1 << (33 - (v16 ^ 0x1F)));
  if ( (int)v17 < 64 )
    v17 = 64;
  if ( (_DWORD)v17 != (_DWORD)v7 )
  {
    v18 = (((4 * (int)v17 / 3u + 1) | ((unsigned __int64)(4 * (int)v17 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v17 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v17 / 3u + 1) >> 1)
        | (((((4 * (int)v17 / 3u + 1) | ((unsigned __int64)(4 * (int)v17 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v17 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v17 / 3u + 1) >> 1)) >> 4);
    v19 = (v18 >> 8) | v18;
    v20 = (v19 | (v19 >> 16)) + 1;
    v21 = 16 * ((v19 | (v19 >> 16)) + 1);
    goto LABEL_25;
  }
  *(_QWORD *)(a1 + 344) = 0;
  v42 = &v15[2 * v17];
  do
  {
    if ( v15 )
      *v15 = -8;
    v15 += 2;
  }
  while ( v42 != v15 );
LABEL_13:
  v10 = *(unsigned __int64 **)(a1 + 424);
  v11 = &v10[2 * *(unsigned int *)(a1 + 432)];
  while ( v10 != v11 )
  {
    v12 = *v10;
    v10 += 2;
    _libc_free(v12);
  }
  v13 = *(unsigned int *)(a1 + 384);
  *(_DWORD *)(a1 + 432) = 0;
  if ( (_DWORD)v13 )
  {
    v36 = *(_QWORD **)(a1 + 376);
    *(_QWORD *)(a1 + 440) = 0;
    v37 = *v36;
    v38 = &v36[v13];
    v39 = v36 + 1;
    *(_QWORD *)(a1 + 360) = v37;
    *(_QWORD *)(a1 + 368) = v37 + 4096;
    while ( v38 != v39 )
    {
      v40 = *v39++;
      _libc_free(v40);
    }
    *(_DWORD *)(a1 + 384) = 1;
  }
}
