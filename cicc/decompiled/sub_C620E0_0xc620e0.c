// Function: sub_C620E0
// Address: 0xc620e0
//
_DWORD *__fastcall sub_C620E0(__int64 a1, _DWORD *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  _DWORD *v6; // r8
  int v7; // r11d
  unsigned int v8; // ecx
  _DWORD *v9; // rax
  int v10; // edi
  int v12; // eax
  int v13; // edx
  int v14; // eax
  int v15; // esi
  __int64 v16; // r9
  unsigned int v17; // ecx
  int v18; // eax
  int v19; // r11d
  _DWORD *v20; // r10
  int v21; // eax
  int v22; // ecx
  __int64 v23; // r9
  int v24; // r11d
  unsigned int v25; // edi
  int v26; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  v8 = (v4 - 1) & (37 * *a2);
  v9 = (_DWORD *)(v5 + ((unsigned __int64)v8 << 7));
  v10 = *v9;
  if ( *a2 == *v9 )
    return v9 + 2;
  while ( v10 != -1 )
  {
    if ( !v6 && v10 == -2 )
      v6 = v9;
    v8 = (v4 - 1) & (v7 + v8);
    v9 = (_DWORD *)(v5 + ((unsigned __int64)v8 << 7));
    v10 = *v9;
    if ( *a2 == *v9 )
      return v9 + 2;
    ++v7;
  }
  if ( !v6 )
    v6 = v9;
  v12 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_18:
    sub_C61E30(a1, 2 * v4);
    v14 = *(_DWORD *)(a1 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v17 = (v14 - 1) & (37 * *a2);
      v6 = (_DWORD *)(v16 + ((unsigned __int64)v17 << 7));
      v18 = *v6;
      if ( *v6 == *a2 )
        goto LABEL_14;
      v19 = 1;
      v20 = 0;
      while ( v18 != -1 )
      {
        if ( !v20 && v18 == -2 )
          v20 = v6;
        v17 = v15 & (v19 + v17);
        v6 = (_DWORD *)(v16 + ((unsigned __int64)v17 << 7));
        v18 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_14;
        ++v19;
      }
LABEL_22:
      if ( v20 )
        v6 = v20;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_C61E30(a1, v4);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v20 = 0;
      v24 = 1;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v25 = (v21 - 1) & (37 * *a2);
      v6 = (_DWORD *)(v23 + ((unsigned __int64)v25 << 7));
      v26 = *v6;
      if ( *a2 == *v6 )
        goto LABEL_14;
      while ( v26 != -1 )
      {
        if ( !v20 && v26 == -2 )
          v20 = v6;
        v25 = v22 & (v24 + v25);
        v6 = (_DWORD *)(v23 + ((unsigned __int64)v25 << 7));
        v26 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_14;
        ++v24;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v6 != -1 )
    --*(_DWORD *)(a1 + 20);
  *v6 = *a2;
  memset(v6 + 2, 0, 0x78u);
  *((_QWORD *)v6 + 4) = v6 + 12;
  *((_QWORD *)v6 + 8) = v6 + 20;
  *((_QWORD *)v6 + 9) = 0x300000000LL;
  return v6 + 2;
}
