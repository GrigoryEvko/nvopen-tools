// Function: sub_1031390
// Address: 0x1031390
//
_QWORD *__fastcall sub_1031390(__int64 a1, __int64 *a2)
{
  unsigned int v4; // r8d
  __int64 v5; // r9
  __int64 v6; // rdi
  _QWORD *v7; // rcx
  int v8; // r11d
  unsigned int v9; // esi
  _QWORD *v10; // rax
  __int64 v11; // rdx
  int v13; // eax
  int v14; // edx
  __int64 v15; // rax
  int v16; // eax
  int v17; // edi
  __int64 v18; // r9
  unsigned int v19; // esi
  __int64 v20; // rax
  int v21; // r11d
  _QWORD *v22; // r10
  int v23; // eax
  int v24; // eax
  __int64 v25; // r8
  int v26; // r11d
  unsigned int v27; // esi
  __int64 v28; // r9

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *a2;
  v7 = 0;
  v8 = 1;
  v9 = (v4 - 1) & (*a2 ^ ((unsigned __int64)*a2 >> 9));
  v10 = (_QWORD *)(v5 + 80LL * v9);
  v11 = *v10;
  if ( v6 == *v10 )
    return v10 + 1;
  while ( v11 != -4 )
  {
    if ( !v7 && v11 == -16 )
      v7 = v10;
    v9 = (v4 - 1) & (v8 + v9);
    v10 = (_QWORD *)(v5 + 80LL * v9);
    v11 = *v10;
    if ( v6 == *v10 )
      return v10 + 1;
    ++v8;
  }
  if ( !v7 )
    v7 = v10;
  v13 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_18:
    sub_1031120(a1, 2 * v4);
    v16 = *(_DWORD *)(a1 + 24);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 8);
      v14 = *(_DWORD *)(a1 + 16) + 1;
      v19 = (v16 - 1) & (*a2 ^ ((unsigned __int64)*a2 >> 9));
      v7 = (_QWORD *)(v18 + 80LL * v19);
      v20 = *v7;
      if ( *v7 == *a2 )
        goto LABEL_14;
      v21 = 1;
      v22 = 0;
      while ( v20 != -4 )
      {
        if ( !v22 && v20 == -16 )
          v22 = v7;
        v19 = v17 & (v21 + v19);
        v7 = (_QWORD *)(v18 + 80LL * v19);
        v20 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_14;
        ++v21;
      }
LABEL_22:
      if ( v22 )
        v7 = v22;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v14 <= v4 >> 3 )
  {
    sub_1031120(a1, v4);
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 8);
      v22 = 0;
      v26 = 1;
      v27 = v24 & (*a2 ^ ((unsigned __int64)*a2 >> 9));
      v7 = (_QWORD *)(v25 + 80LL * v27);
      v14 = *(_DWORD *)(a1 + 16) + 1;
      v28 = *v7;
      if ( *a2 == *v7 )
        goto LABEL_14;
      while ( v28 != -4 )
      {
        if ( !v22 && v28 == -16 )
          v22 = v7;
        v27 = v24 & (v26 + v27);
        v7 = (_QWORD *)(v25 + 80LL * v27);
        v28 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_14;
        ++v26;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *v7 != -4 )
    --*(_DWORD *)(a1 + 20);
  v15 = *a2;
  v7[9] = 0;
  *(_OWORD *)(v7 + 5) = 0;
  *v7 = v15;
  v7[5] = 0xBFFFFFFFFFFFFFFELL;
  *(_OWORD *)(v7 + 1) = 0;
  *(_OWORD *)(v7 + 3) = 0;
  *(_OWORD *)(v7 + 7) = 0;
  return v7 + 1;
}
