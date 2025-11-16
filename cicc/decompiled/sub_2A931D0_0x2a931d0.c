// Function: sub_2A931D0
// Address: 0x2a931d0
//
__int64 __fastcall sub_2A931D0(__int64 *a1, unsigned int a2)
{
  __int64 v3; // r13
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 *v8; // r10
  int v9; // r14d
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rdx
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // rdi
  int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // ecx
  __int64 v22; // rax
  int v23; // r11d
  __int64 *v24; // r9
  int v25; // eax
  __int64 v26; // rsi
  int v27; // eax
  __int64 v28; // rdi
  int v29; // r11d
  unsigned int v30; // ecx
  __int64 v31; // r8

  v3 = *a1;
  v5 = *(_DWORD *)(*a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)v3;
    goto LABEL_18;
  }
  v6 = a1[1];
  v7 = *(_QWORD *)(v3 + 8);
  v8 = 0;
  v9 = 1;
  v10 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v11 = (__int64 *)(v7 + 16LL * v10);
  v12 = *v11;
  if ( v6 == *v11 )
  {
LABEL_3:
    *((_BYTE *)v11 + 8) = a2;
    return a2;
  }
  while ( v12 != -4096 )
  {
    if ( !v8 && v12 == -8192 )
      v8 = v11;
    v10 = (v5 - 1) & (v9 + v10);
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( v6 == *v11 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v8 )
    v8 = v11;
  v14 = *(_DWORD *)(v3 + 16);
  ++*(_QWORD *)v3;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_18:
    sub_2A92FF0(v3, 2 * v5);
    v17 = *(_DWORD *)(v3 + 24);
    if ( v17 )
    {
      v18 = a1[1];
      v19 = v17 - 1;
      v20 = *(_QWORD *)(v3 + 8);
      v15 = *(_DWORD *)(v3 + 16) + 1;
      v21 = (v17 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v8 = (__int64 *)(v20 + 16LL * v21);
      v22 = *v8;
      if ( *v8 == v18 )
        goto LABEL_14;
      v23 = 1;
      v24 = 0;
      while ( v22 != -4096 )
      {
        if ( !v24 && v22 == -8192 )
          v24 = v8;
        v21 = v19 & (v23 + v21);
        v8 = (__int64 *)(v20 + 16LL * v21);
        v22 = *v8;
        if ( v18 == *v8 )
          goto LABEL_14;
        ++v23;
      }
LABEL_22:
      if ( v24 )
        v8 = v24;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(v3 + 16);
    BUG();
  }
  if ( v5 - *(_DWORD *)(v3 + 20) - v15 <= v5 >> 3 )
  {
    sub_2A92FF0(v3, v5);
    v25 = *(_DWORD *)(v3 + 24);
    if ( v25 )
    {
      v26 = a1[1];
      v27 = v25 - 1;
      v28 = *(_QWORD *)(v3 + 8);
      v24 = 0;
      v29 = 1;
      v30 = v27 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v15 = *(_DWORD *)(v3 + 16) + 1;
      v8 = (__int64 *)(v28 + 16LL * v30);
      v31 = *v8;
      if ( v26 == *v8 )
        goto LABEL_14;
      while ( v31 != -4096 )
      {
        if ( !v24 && v31 == -8192 )
          v24 = v8;
        v30 = v27 & (v29 + v30);
        v8 = (__int64 *)(v28 + 16LL * v30);
        v31 = *v8;
        if ( v26 == *v8 )
          goto LABEL_14;
        ++v29;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(v3 + 16) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(v3 + 20);
  v16 = a1[1];
  *((_BYTE *)v8 + 8) = 0;
  *v8 = v16;
  *((_BYTE *)v8 + 8) = a2;
  return a2;
}
