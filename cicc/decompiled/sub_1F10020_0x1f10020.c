// Function: sub_1F10020
// Address: 0x1f10020
//
void __fastcall sub_1F10020(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  unsigned __int64 *v6; // rax
  unsigned __int64 *v7; // rcx
  unsigned __int64 v8; // rsi
  unsigned __int64 *v9; // r12
  unsigned __int64 *v10; // r13
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  unsigned int v13; // ecx
  _QWORD *v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  int v19; // r13d
  __int64 v20; // r12
  _QWORD *v21; // rax
  __int64 v22; // rdx
  _QWORD *j; // rdx
  _QWORD *v24; // r12
  __int64 v25; // rdx
  unsigned __int64 *v26; // r13
  unsigned __int64 *v27; // r12
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rax

  v2 = *(_DWORD *)(a1 + 376);
  ++*(_QWORD *)(a1 + 360);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 380) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 384);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 368));
      *(_QWORD *)(a1 + 368) = 0;
      *(_QWORD *)(a1 + 376) = 0;
      *(_DWORD *)(a1 + 384) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v13 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 384);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v13 = 64;
  if ( v13 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 368);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 376) = 0;
    goto LABEL_7;
  }
  v14 = *(_QWORD **)(a1 + 368);
  v15 = v2 - 1;
  if ( !v15 )
  {
    v20 = 2048;
    v19 = 128;
LABEL_21:
    j___libc_free_0(v14);
    *(_DWORD *)(a1 + 384) = v19;
    v21 = (_QWORD *)sub_22077B0(v20);
    v22 = *(unsigned int *)(a1 + 384);
    *(_QWORD *)(a1 + 376) = 0;
    *(_QWORD *)(a1 + 368) = v21;
    for ( j = &v21[2 * v22]; j != v21; v21 += 2 )
    {
      if ( v21 )
        *v21 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v15, v15);
  v16 = (unsigned int)(1 << (33 - (v15 ^ 0x1F)));
  if ( (int)v16 < 64 )
    v16 = 64;
  if ( (_DWORD)v16 != (_DWORD)v3 )
  {
    v17 = (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v16 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)
        | (((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v16 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4);
    v18 = (v17 >> 8) | v17;
    v19 = (v18 | (v18 >> 16)) + 1;
    v20 = 16 * ((v18 | (v18 >> 16)) + 1);
    goto LABEL_21;
  }
  *(_QWORD *)(a1 + 376) = 0;
  v29 = &v14[2 * v16];
  do
  {
    if ( v14 )
      *v14 = -8;
    v14 += 2;
  }
  while ( v29 != v14 );
LABEL_7:
  v6 = *(unsigned __int64 **)(a1 + 344);
  *(_DWORD *)(a1 + 400) = 0;
  for ( *(_DWORD *)(a1 + 544) = 0; (unsigned __int64 *)(a1 + 336) != v6; *v7 &= 7u )
  {
    v7 = v6;
    v6 = (unsigned __int64 *)v6[1];
    v8 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
    *v6 = v8 | *v6 & 7;
    *(_QWORD *)(v8 + 8) = v6;
    v7[1] = 0;
  }
  v9 = *(unsigned __int64 **)(a1 + 296);
  v10 = &v9[2 * *(unsigned int *)(a1 + 304)];
  while ( v10 != v9 )
  {
    v11 = *v9;
    v9 += 2;
    _libc_free(v11);
  }
  *(_DWORD *)(a1 + 304) = 0;
  v12 = *(unsigned int *)(a1 + 256);
  if ( (_DWORD)v12 )
  {
    v24 = *(_QWORD **)(a1 + 248);
    *(_QWORD *)(a1 + 312) = 0;
    v25 = *v24;
    v26 = &v24[v12];
    v27 = v24 + 1;
    *(_QWORD *)(a1 + 232) = v25;
    *(_QWORD *)(a1 + 240) = v25 + 4096;
    while ( v26 != v27 )
    {
      v28 = *v27++;
      _libc_free(v28);
    }
    *(_DWORD *)(a1 + 256) = 1;
  }
}
