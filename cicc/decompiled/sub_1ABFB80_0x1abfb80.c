// Function: sub_1ABFB80
// Address: 0x1abfb80
//
__int64 __fastcall sub_1ABFB80(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  __int64 *v6; // r10
  int v7; // r11d
  unsigned int v8; // eax
  __int64 *v9; // rdi
  __int64 v10; // rcx
  int v12; // eax
  int v13; // edx
  __int64 v14; // rax
  _BYTE *v15; // rsi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // r8
  unsigned int v19; // eax
  __int64 v20; // rdi
  int v21; // r11d
  __int64 *v22; // r9
  int v23; // eax
  int v24; // ecx
  __int64 v25; // r8
  int v26; // r11d
  unsigned int v27; // eax
  __int64 v28; // rdi

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  v8 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v9 = (__int64 *)(v5 + 8LL * v8);
  v10 = *v9;
  if ( *v9 == *a2 )
    return 0;
  while ( v10 != -8 )
  {
    if ( v10 != -16 || v6 )
      v9 = v6;
    v8 = (v4 - 1) & (v7 + v8);
    v10 = *(_QWORD *)(v5 + 8LL * v8);
    if ( *a2 == v10 )
      return 0;
    ++v7;
    v6 = v9;
    v9 = (__int64 *)(v5 + 8LL * v8);
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( !v6 )
    v6 = v9;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_21:
    sub_13B3D40(a1, 2 * v4);
    v16 = *(_DWORD *)(a1 + 24);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 8);
      v19 = (v16 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v6 = (__int64 *)(v18 + 8LL * v19);
      v20 = *v6;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v6 == *a2 )
        goto LABEL_13;
      v21 = 1;
      v22 = 0;
      while ( v20 != -8 )
      {
        if ( v20 == -16 && !v22 )
          v22 = v6;
        v19 = v17 & (v21 + v19);
        v6 = (__int64 *)(v18 + 8LL * v19);
        v20 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_13;
        ++v21;
      }
LABEL_25:
      if ( v22 )
        v6 = v22;
      goto LABEL_13;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_13B3D40(a1, v4);
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 8);
      v22 = 0;
      v26 = 1;
      v27 = (v23 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v6 = (__int64 *)(v25 + 8LL * v27);
      v28 = *v6;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v6 == *a2 )
        goto LABEL_13;
      while ( v28 != -8 )
      {
        if ( !v22 && v28 == -16 )
          v22 = v6;
        v27 = v24 & (v26 + v27);
        v6 = (__int64 *)(v25 + 8LL * v27);
        v28 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_13;
        ++v26;
      }
      goto LABEL_25;
    }
    goto LABEL_42;
  }
LABEL_13:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v6 != -8 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  *v6 = *a2;
  v15 = *(_BYTE **)(a1 + 40);
  if ( v15 == *(_BYTE **)(a1 + 48) )
  {
    sub_1292090(a1 + 32, v15, a2);
    return 1;
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v14;
      v15 = *(_BYTE **)(a1 + 40);
    }
    *(_QWORD *)(a1 + 40) = v15 + 8;
    return 1;
  }
}
