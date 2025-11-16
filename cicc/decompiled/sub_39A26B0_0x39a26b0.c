// Function: sub_39A26B0
// Address: 0x39a26b0
//
void __fastcall sub_39A26B0(__int64 a1)
{
  __int64 v1; // r9
  __int64 v2; // rsi
  __int64 i; // r10
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  _QWORD *v9; // rcx
  int v10; // eax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *j; // rdx
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // r12
  unsigned __int64 v17; // rdi
  unsigned int v18; // ecx
  _QWORD *v19; // rdi
  unsigned int v20; // eax
  int v21; // eax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  int v24; // r13d
  unsigned __int64 v25; // r12
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *k; // rdx
  _QWORD *v29; // rax

  v1 = a1 + 8;
  v2 = *(_QWORD *)(a1 + 368);
  for ( i = *(_QWORD *)(a1 + 376); i != v2; v2 += 88 )
  {
    while ( 1 )
    {
      v5 = *(unsigned int *)(v2 + 16);
      if ( (_DWORD)v5 )
        break;
      v2 += 88;
      if ( i == v2 )
        goto LABEL_10;
    }
    v6 = 8 * v5;
    v7 = 0;
    do
    {
      v8 = *(_QWORD *)(*(_QWORD *)(v2 + 8) + v7);
      *(_QWORD *)(v8 + 40) = v1;
      v9 = *(_QWORD **)(a1 + 40);
      if ( v9 )
      {
        *(_QWORD *)v8 = *v9;
        **(_QWORD **)(a1 + 40) = v8 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v7 += 8;
      *(_QWORD *)(a1 + 40) = v8;
    }
    while ( v6 != v7 );
    *(_DWORD *)(v2 + 16) = 0;
  }
LABEL_10:
  v10 = *(_DWORD *)(a1 + 352);
  ++*(_QWORD *)(a1 + 336);
  if ( !v10 )
  {
    if ( !*(_DWORD *)(a1 + 356) )
      goto LABEL_16;
    v11 = *(unsigned int *)(a1 + 360);
    if ( (unsigned int)v11 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 344));
      *(_QWORD *)(a1 + 344) = 0;
      *(_QWORD *)(a1 + 352) = 0;
      *(_DWORD *)(a1 + 360) = 0;
      goto LABEL_16;
    }
    goto LABEL_13;
  }
  v18 = 4 * v10;
  v11 = *(unsigned int *)(a1 + 360);
  if ( (unsigned int)(4 * v10) < 0x40 )
    v18 = 64;
  if ( (unsigned int)v11 <= v18 )
  {
LABEL_13:
    v12 = *(_QWORD **)(a1 + 344);
    for ( j = &v12[2 * v11]; j != v12; v12 += 2 )
      *v12 = -8;
    *(_QWORD *)(a1 + 352) = 0;
    goto LABEL_16;
  }
  v19 = *(_QWORD **)(a1 + 344);
  v20 = v10 - 1;
  if ( !v20 )
  {
    v25 = 2048;
    v24 = 128;
LABEL_31:
    j___libc_free_0((unsigned __int64)v19);
    *(_DWORD *)(a1 + 360) = v24;
    v26 = (_QWORD *)sub_22077B0(v25);
    v27 = *(unsigned int *)(a1 + 360);
    *(_QWORD *)(a1 + 352) = 0;
    *(_QWORD *)(a1 + 344) = v26;
    for ( k = &v26[2 * v27]; k != v26; v26 += 2 )
    {
      if ( v26 )
        *v26 = -8;
    }
    goto LABEL_16;
  }
  _BitScanReverse(&v20, v20);
  v21 = 1 << (33 - (v20 ^ 0x1F));
  if ( v21 < 64 )
    v21 = 64;
  if ( (_DWORD)v11 != v21 )
  {
    v22 = (4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1);
    v23 = ((v22 | (v22 >> 2)) >> 4) | v22 | (v22 >> 2) | ((((v22 | (v22 >> 2)) >> 4) | v22 | (v22 >> 2)) >> 8);
    v24 = (v23 | (v23 >> 16)) + 1;
    v25 = 16 * ((v23 | (v23 >> 16)) + 1);
    goto LABEL_31;
  }
  *(_QWORD *)(a1 + 352) = 0;
  v29 = &v19[2 * (unsigned int)v11];
  do
  {
    if ( v19 )
      *v19 = -8;
    v19 += 2;
  }
  while ( v29 != v19 );
LABEL_16:
  v14 = *(_QWORD *)(a1 + 368);
  v15 = *(_QWORD *)(a1 + 376);
  if ( v14 != v15 )
  {
    v16 = *(_QWORD *)(a1 + 368);
    do
    {
      v17 = *(_QWORD *)(v16 + 8);
      if ( v17 != v16 + 24 )
        _libc_free(v17);
      v16 += 88;
    }
    while ( v15 != v16 );
    *(_QWORD *)(a1 + 376) = v14;
  }
}
