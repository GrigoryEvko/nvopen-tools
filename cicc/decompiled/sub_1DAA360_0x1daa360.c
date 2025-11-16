// Function: sub_1DAA360
// Address: 0x1daa360
//
void __fastcall sub_1DAA360(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rcx
  unsigned __int64 v7; // r8
  unsigned __int64 v8; // r9
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // rsi
  int v12; // eax
  unsigned int v13; // ecx
  __int64 v14; // rdx
  _DWORD *v15; // rax
  _DWORD *i; // rdx
  int v17; // eax
  __int64 v18; // rdx
  _DWORD *v19; // rax
  unsigned int v20; // ecx
  _QWORD *v21; // rax
  _QWORD *m; // rdx
  _QWORD *v23; // rdi
  unsigned int v24; // eax
  int v25; // eax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  int v28; // ebx
  __int64 v29; // r12
  _QWORD *v30; // rax
  __int64 v31; // rdx
  _QWORD *k; // rdx
  _DWORD *v33; // rdi
  unsigned int v34; // eax
  int v35; // eax
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rax
  int v38; // ebx
  __int64 v39; // r12
  _DWORD *v40; // rax
  __int64 v41; // rdx
  _DWORD *j; // rdx
  _DWORD *v43; // rax
  _QWORD *v44; // rax

  v1 = *(_QWORD *)(a1 + 232);
  if ( !v1 )
    return;
  v2 = *(_QWORD *)(v1 + 152);
  v3 = *(unsigned int *)(v1 + 160);
  *(_QWORD *)(v1 + 120) = 0;
  v4 = v2 + 8 * v3;
  while ( v2 != v4 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(v4 - 8);
      v4 -= 8;
      if ( !v5 )
        break;
      sub_1DA8360(*(_QWORD *)(v5 + 360));
      v9 = *(_QWORD *)(v5 + 312);
      if ( v9 != v5 + 328 )
        _libc_free(v9);
      if ( *(_DWORD *)(v5 + 296) )
      {
        sub_1DA9BF0(v5 + 216, (char *)sub_1DA8010, 0, v6, v7, v8);
        v19 = (_DWORD *)(v5 + 280);
        do
          *v19++ = 0;
        while ( v19 != (_DWORD *)(v5 + 296) );
      }
      v10 = *(_QWORD *)(v5 + 40);
      if ( v10 != v5 + 56 )
        _libc_free(v10);
      v11 = *(_QWORD *)(v5 + 16);
      if ( v11 )
        sub_161E7C0(v5 + 16, v11);
      j_j___libc_free_0(v5, 392);
      if ( v2 == v4 )
        goto LABEL_13;
    }
  }
LABEL_13:
  v12 = *(_DWORD *)(v1 + 248);
  ++*(_QWORD *)(v1 + 232);
  *(_DWORD *)(v1 + 160) = 0;
  if ( !v12 )
  {
    if ( !*(_DWORD *)(v1 + 252) )
      goto LABEL_20;
    v14 = *(unsigned int *)(v1 + 256);
    if ( (unsigned int)v14 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(v1 + 240));
      *(_QWORD *)(v1 + 240) = 0;
      *(_QWORD *)(v1 + 248) = 0;
      *(_DWORD *)(v1 + 256) = 0;
      goto LABEL_20;
    }
    goto LABEL_17;
  }
  v13 = 4 * v12;
  v14 = *(unsigned int *)(v1 + 256);
  if ( (unsigned int)(4 * v12) < 0x40 )
    v13 = 64;
  if ( (unsigned int)v14 <= v13 )
  {
LABEL_17:
    v15 = *(_DWORD **)(v1 + 240);
    for ( i = &v15[4 * v14]; i != v15; v15 += 4 )
      *v15 = -1;
    *(_QWORD *)(v1 + 248) = 0;
    goto LABEL_20;
  }
  v33 = *(_DWORD **)(v1 + 240);
  v34 = v12 - 1;
  if ( !v34 )
  {
    v39 = 2048;
    v38 = 128;
LABEL_53:
    j___libc_free_0(v33);
    *(_DWORD *)(v1 + 256) = v38;
    v40 = (_DWORD *)sub_22077B0(v39);
    v41 = *(unsigned int *)(v1 + 256);
    *(_QWORD *)(v1 + 248) = 0;
    *(_QWORD *)(v1 + 240) = v40;
    for ( j = &v40[4 * v41]; j != v40; v40 += 4 )
    {
      if ( v40 )
        *v40 = -1;
    }
    goto LABEL_20;
  }
  _BitScanReverse(&v34, v34);
  v35 = 1 << (33 - (v34 ^ 0x1F));
  if ( v35 < 64 )
    v35 = 64;
  if ( (_DWORD)v14 != v35 )
  {
    v36 = (((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
        | (4 * v35 / 3u + 1)
        | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)
        | (((((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
          | (4 * v35 / 3u + 1)
          | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 4);
    v37 = (v36 >> 8) | v36;
    v38 = (v37 | (v37 >> 16)) + 1;
    v39 = 16 * ((v37 | (v37 >> 16)) + 1);
    goto LABEL_53;
  }
  *(_QWORD *)(v1 + 248) = 0;
  v43 = &v33[4 * (unsigned int)v14];
  do
  {
    if ( v33 )
      *v33 = -1;
    v33 += 4;
  }
  while ( v43 != v33 );
LABEL_20:
  v17 = *(_DWORD *)(v1 + 280);
  ++*(_QWORD *)(v1 + 264);
  if ( !v17 )
  {
    if ( *(_DWORD *)(v1 + 284) )
    {
      v18 = *(unsigned int *)(v1 + 288);
      if ( (unsigned int)v18 > 0x40 )
      {
        j___libc_free_0(*(_QWORD *)(v1 + 272));
        *(_QWORD *)(v1 + 272) = 0;
        *(_QWORD *)(v1 + 280) = 0;
        *(_DWORD *)(v1 + 288) = 0;
        goto LABEL_24;
      }
      goto LABEL_35;
    }
LABEL_24:
    *(_WORD *)(v1 + 144) = 0;
    return;
  }
  v20 = 4 * v17;
  v18 = *(unsigned int *)(v1 + 288);
  if ( (unsigned int)(4 * v17) < 0x40 )
    v20 = 64;
  if ( v20 < (unsigned int)v18 )
  {
    v23 = *(_QWORD **)(v1 + 272);
    v24 = v17 - 1;
    if ( v24 )
    {
      _BitScanReverse(&v24, v24);
      v25 = 1 << (33 - (v24 ^ 0x1F));
      if ( v25 < 64 )
        v25 = 64;
      if ( (_DWORD)v18 == v25 )
      {
        *(_QWORD *)(v1 + 280) = 0;
        v44 = &v23[2 * (unsigned int)v18];
        do
        {
          if ( v23 )
            *v23 = -8;
          v23 += 2;
        }
        while ( v44 != v23 );
        goto LABEL_24;
      }
      v26 = (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
          | (4 * v25 / 3u + 1)
          | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)
          | (((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
            | (4 * v25 / 3u + 1)
            | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4);
      v27 = (v26 >> 8) | v26;
      v28 = (v27 | (v27 >> 16)) + 1;
      v29 = 16 * ((v27 | (v27 >> 16)) + 1);
    }
    else
    {
      v29 = 2048;
      v28 = 128;
    }
    j___libc_free_0(v23);
    *(_DWORD *)(v1 + 288) = v28;
    v30 = (_QWORD *)sub_22077B0(v29);
    v31 = *(unsigned int *)(v1 + 288);
    *(_QWORD *)(v1 + 280) = 0;
    *(_QWORD *)(v1 + 272) = v30;
    for ( k = &v30[2 * v31]; k != v30; v30 += 2 )
    {
      if ( v30 )
        *v30 = -8;
    }
    goto LABEL_24;
  }
LABEL_35:
  v21 = *(_QWORD **)(v1 + 272);
  for ( m = &v21[2 * v18]; m != v21; v21 += 2 )
    *v21 = -8;
  *(_QWORD *)(v1 + 280) = 0;
  *(_WORD *)(v1 + 144) = 0;
}
