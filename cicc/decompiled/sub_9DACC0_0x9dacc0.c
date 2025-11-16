// Function: sub_9DACC0
// Address: 0x9dacc0
//
_DWORD *__fastcall sub_9DACC0(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  unsigned int v5; // eax
  int v7; // eax
  unsigned int v8; // esi
  __int64 v10; // r9
  int v11; // edx
  unsigned int v12; // esi
  int v13; // edi
  int v14; // r11d
  _DWORD *v15; // r10
  int v16; // esi
  __int64 v17; // r9
  int v18; // esi
  int v19; // r11d
  unsigned int v20; // edx
  int v21; // edi
  int v22; // eax

  v5 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v7 = (v5 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
  {
    v8 = *(_DWORD *)(a1 + 24);
    if ( 4 * v7 < 3 * v8 )
      goto LABEL_3;
LABEL_8:
    sub_9DA8C0(a1, 2 * v8);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v10 = a1 + 16;
      v11 = 3;
    }
    else
    {
      v22 = *(_DWORD *)(a1 + 24);
      v10 = *(_QWORD *)(a1 + 16);
      if ( !v22 )
        goto LABEL_34;
      v11 = v22 - 1;
    }
    v12 = v11 & (37 * *a2);
    v7 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
    a3 = (_DWORD *)(v10 + 16LL * v12);
    v13 = *a3;
    if ( *a2 == *a3 )
      goto LABEL_4;
    v14 = 1;
    v15 = 0;
    while ( v13 != -1 )
    {
      if ( v13 == -2 && !v15 )
        v15 = a3;
      v12 = v11 & (v14 + v12);
      a3 = (_DWORD *)(v10 + 16LL * v12);
      v13 = *a3;
      if ( *a2 == *a3 )
        goto LABEL_4;
      ++v14;
    }
    goto LABEL_13;
  }
  v8 = 4;
  if ( (unsigned int)(4 * v7) >= 0xC )
    goto LABEL_8;
LABEL_3:
  if ( v8 - (v7 + *(_DWORD *)(a1 + 12)) > v8 >> 3 )
    goto LABEL_4;
  sub_9DA8C0(a1, v8);
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v17 = a1 + 16;
    v18 = 3;
    goto LABEL_19;
  }
  v16 = *(_DWORD *)(a1 + 24);
  v17 = *(_QWORD *)(a1 + 16);
  if ( !v16 )
  {
LABEL_34:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
  v18 = v16 - 1;
LABEL_19:
  v19 = 1;
  v15 = 0;
  v20 = v18 & (37 * *a2);
  v7 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
  a3 = (_DWORD *)(v17 + 16LL * v20);
  v21 = *a3;
  if ( *a2 == *a3 )
    goto LABEL_4;
  while ( v21 != -1 )
  {
    if ( v21 == -2 && !v15 )
      v15 = a3;
    v20 = v18 & (v19 + v20);
    a3 = (_DWORD *)(v17 + 16LL * v20);
    v21 = *a3;
    if ( *a2 == *a3 )
      goto LABEL_4;
    ++v19;
  }
LABEL_13:
  if ( v15 )
    a3 = v15;
LABEL_4:
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v7);
  if ( *a3 != -1 )
    --*(_DWORD *)(a1 + 12);
  return a3;
}
