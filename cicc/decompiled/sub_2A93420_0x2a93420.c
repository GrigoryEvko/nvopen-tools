// Function: sub_2A93420
// Address: 0x2a93420
//
__int64 __fastcall sub_2A93420(__int64 *a1)
{
  __int64 v2; // r12
  unsigned int v3; // esi
  __int64 v4; // rdi
  __int64 v5; // r8
  __int64 *v6; // r10
  int v7; // r11d
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // rdx
  int v12; // eax
  int v13; // edx
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rdi
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // ecx
  __int64 v20; // rax
  int v21; // r11d
  __int64 *v22; // r9
  int v23; // eax
  __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // r8
  int v27; // r11d
  unsigned int v28; // ecx
  __int64 v29; // rdi

  v2 = *a1;
  v3 = *(_DWORD *)(*a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)v2;
    goto LABEL_18;
  }
  v4 = a1[1];
  v5 = *(_QWORD *)(v2 + 8);
  v6 = 0;
  v7 = 1;
  v8 = (v3 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v9 = (__int64 *)(v5 + 16LL * v8);
  v10 = *v9;
  if ( v4 == *v9 )
  {
LABEL_3:
    *((_BYTE *)v9 + 8) = 0;
    return 0;
  }
  while ( v10 != -4096 )
  {
    if ( !v6 && v10 == -8192 )
      v6 = v9;
    v8 = (v3 - 1) & (v7 + v8);
    v9 = (__int64 *)(v5 + 16LL * v8);
    v10 = *v9;
    if ( v4 == *v9 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v6 )
    v6 = v9;
  v12 = *(_DWORD *)(v2 + 16);
  ++*(_QWORD *)v2;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v3 )
  {
LABEL_18:
    sub_2A92FF0(v2, 2 * v3);
    v15 = *(_DWORD *)(v2 + 24);
    if ( v15 )
    {
      v16 = a1[1];
      v17 = v15 - 1;
      v18 = *(_QWORD *)(v2 + 8);
      v13 = *(_DWORD *)(v2 + 16) + 1;
      v19 = (v15 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v6 = (__int64 *)(v18 + 16LL * v19);
      v20 = *v6;
      if ( *v6 == v16 )
        goto LABEL_14;
      v21 = 1;
      v22 = 0;
      while ( v20 != -4096 )
      {
        if ( !v22 && v20 == -8192 )
          v22 = v6;
        v19 = v17 & (v21 + v19);
        v6 = (__int64 *)(v18 + 16LL * v19);
        v20 = *v6;
        if ( v16 == *v6 )
          goto LABEL_14;
        ++v21;
      }
LABEL_22:
      if ( v22 )
        v6 = v22;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(v2 + 16);
    BUG();
  }
  if ( v3 - *(_DWORD *)(v2 + 20) - v13 <= v3 >> 3 )
  {
    sub_2A92FF0(v2, v3);
    v23 = *(_DWORD *)(v2 + 24);
    if ( v23 )
    {
      v24 = a1[1];
      v25 = v23 - 1;
      v26 = *(_QWORD *)(v2 + 8);
      v22 = 0;
      v27 = 1;
      v28 = v25 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v13 = *(_DWORD *)(v2 + 16) + 1;
      v6 = (__int64 *)(v26 + 16LL * v28);
      v29 = *v6;
      if ( v24 == *v6 )
        goto LABEL_14;
      while ( v29 != -4096 )
      {
        if ( !v22 && v29 == -8192 )
          v22 = v6;
        v28 = v25 & (v27 + v28);
        v6 = (__int64 *)(v26 + 16LL * v28);
        v29 = *v6;
        if ( v24 == *v6 )
          goto LABEL_14;
        ++v27;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(v2 + 16) = v13;
  if ( *v6 != -4096 )
    --*(_DWORD *)(v2 + 20);
  v14 = a1[1];
  *((_BYTE *)v6 + 8) = 0;
  *v6 = v14;
  *((_BYTE *)v6 + 8) = 0;
  return 0;
}
