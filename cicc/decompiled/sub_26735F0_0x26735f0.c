// Function: sub_26735F0
// Address: 0x26735f0
//
_QWORD *__fastcall sub_26735F0(__int64 a1, _QWORD *a2)
{
  unsigned int v3; // esi
  __int64 v4; // r8
  _QWORD *v5; // rdx
  int v6; // r11d
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  __int64 v9; // r10
  int v11; // eax
  int v12; // ecx
  int v13; // eax
  int v14; // eax
  __int64 v15; // r9
  unsigned int v16; // esi
  __int64 v17; // r8
  int v18; // r11d
  _QWORD *v19; // r10
  int v20; // eax
  int v21; // eax
  __int64 v22; // r9
  int v23; // r11d
  unsigned int v24; // esi
  __int64 v25; // r8

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = 0;
  v6 = 1;
  v7 = (v3 - 1) & (*a2 ^ (*a2 >> 9));
  v8 = (_QWORD *)(v4 + ((unsigned __int64)v7 << 7));
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  while ( v9 != -4 )
  {
    if ( !v5 && v9 == -16 )
      v5 = v8;
    v7 = (v3 - 1) & (v6 + v7);
    v8 = (_QWORD *)(v4 + ((unsigned __int64)v7 << 7));
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v6;
  }
  if ( !v5 )
    v5 = v8;
  v11 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v3 )
  {
LABEL_18:
    sub_2673380(a1, 2 * v3);
    v13 = *(_DWORD *)(a1 + 24);
    if ( v13 )
    {
      v14 = v13 - 1;
      v15 = *(_QWORD *)(a1 + 8);
      v12 = *(_DWORD *)(a1 + 16) + 1;
      v16 = v14 & (*a2 ^ (*a2 >> 9));
      v5 = (_QWORD *)(v15 + ((unsigned __int64)v16 << 7));
      v17 = *v5;
      if ( *v5 == *a2 )
        goto LABEL_14;
      v18 = 1;
      v19 = 0;
      while ( v17 != -4 )
      {
        if ( !v19 && v17 == -16 )
          v19 = v5;
        v16 = v14 & (v18 + v16);
        v5 = (_QWORD *)(v15 + ((unsigned __int64)v16 << 7));
        v17 = *v5;
        if ( *a2 == *v5 )
          goto LABEL_14;
        ++v18;
      }
LABEL_22:
      if ( v19 )
        v5 = v19;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v3 - *(_DWORD *)(a1 + 20) - v12 <= v3 >> 3 )
  {
    sub_2673380(a1, v3);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 8);
      v19 = 0;
      v23 = 1;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      v24 = v21 & (*a2 ^ (*a2 >> 9));
      v5 = (_QWORD *)(v22 + ((unsigned __int64)v24 << 7));
      v25 = *v5;
      if ( *a2 == *v5 )
        goto LABEL_14;
      while ( v25 != -4 )
      {
        if ( !v19 && v25 == -16 )
          v19 = v5;
        v24 = v21 & (v23 + v24);
        v5 = (_QWORD *)(v22 + ((unsigned __int64)v24 << 7));
        v25 = *v5;
        if ( *a2 == *v5 )
          goto LABEL_14;
        ++v23;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v5 != -4 )
    --*(_DWORD *)(a1 + 20);
  *v5 = *a2;
  memset(v5 + 1, 0, 0x78u);
  *((_DWORD *)v5 + 2) = 65793;
  v5[3] = v5 + 6;
  v5[9] = v5 + 12;
  v5[4] = 2;
  *((_BYTE *)v5 + 44) = 1;
  v5[10] = 4;
  *((_BYTE *)v5 + 92) = 1;
  return v5 + 1;
}
