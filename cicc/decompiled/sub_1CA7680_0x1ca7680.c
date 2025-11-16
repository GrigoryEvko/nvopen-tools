// Function: sub_1CA7680
// Address: 0x1ca7680
//
__int64 __fastcall sub_1CA7680(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 *v5; // rax
  __int64 v7; // rbx
  unsigned int v8; // esi
  __int64 v9; // r8
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // rcx
  int v13; // r10d
  __int64 *v14; // rdx
  int v15; // eax
  int v16; // ecx
  int v17; // eax
  int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // eax
  __int64 v21; // rdi
  int v22; // r10d
  __int64 *v23; // r9
  int v24; // eax
  int v25; // eax
  __int64 v26; // rdi
  int v27; // r9d
  unsigned int v28; // r13d
  __int64 *v29; // r8
  __int64 v30; // rsi
  __int64 *v31; // r11
  unsigned int v32[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(__int64 **)a1;
  v32[0] = 0;
  if ( (unsigned __int8)sub_1C98370(*(_QWORD **)(a1 + 8), *v5, a2, v32) )
  {
    *a3 = 1;
    return v32[0];
  }
  *a3 = 0;
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_DWORD *)(v7 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)v7;
    goto LABEL_16;
  }
  v9 = *(_QWORD *)(v7 + 8);
  v10 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (__int64 *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( *v11 == a2 )
    return *((unsigned int *)v11 + 2);
  v13 = 1;
  v14 = 0;
  while ( v12 != -8 )
  {
    if ( v14 || v12 != -16 )
      v11 = v14;
    v10 = (v8 - 1) & (v13 + v10);
    v31 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v31;
    if ( *v31 == a2 )
      return *((unsigned int *)v31 + 2);
    ++v13;
    v14 = v11;
    v11 = (__int64 *)(v9 + 16LL * v10);
  }
  if ( !v14 )
    v14 = v11;
  v15 = *(_DWORD *)(v7 + 16);
  ++*(_QWORD *)v7;
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v8 )
  {
LABEL_16:
    sub_177C7D0(v7, 2 * v8);
    v17 = *(_DWORD *)(v7 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v7 + 8);
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(v7 + 16) + 1;
      v14 = (__int64 *)(v19 + 16LL * v20);
      v21 = *v14;
      if ( *v14 != a2 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -8 )
        {
          if ( !v23 && v21 == -16 )
            v23 = v14;
          v20 = v18 & (v22 + v20);
          v14 = (__int64 *)(v19 + 16LL * v20);
          v21 = *v14;
          if ( *v14 == a2 )
            goto LABEL_12;
          ++v22;
        }
        if ( v23 )
          v14 = v23;
      }
      goto LABEL_12;
    }
    goto LABEL_45;
  }
  if ( v8 - *(_DWORD *)(v7 + 20) - v16 <= v8 >> 3 )
  {
    sub_177C7D0(v7, v8);
    v24 = *(_DWORD *)(v7 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(v7 + 8);
      v27 = 1;
      v28 = v25 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v29 = 0;
      v16 = *(_DWORD *)(v7 + 16) + 1;
      v14 = (__int64 *)(v26 + 16LL * v28);
      v30 = *v14;
      if ( *v14 != a2 )
      {
        while ( v30 != -8 )
        {
          if ( !v29 && v30 == -16 )
            v29 = v14;
          v28 = v25 & (v27 + v28);
          v14 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v14;
          if ( *v14 == a2 )
            goto LABEL_12;
          ++v27;
        }
        if ( v29 )
          v14 = v29;
      }
      goto LABEL_12;
    }
LABEL_45:
    ++*(_DWORD *)(v7 + 16);
    BUG();
  }
LABEL_12:
  *(_DWORD *)(v7 + 16) = v16;
  if ( *v14 != -8 )
    --*(_DWORD *)(v7 + 20);
  *v14 = a2;
  *((_DWORD *)v14 + 2) = 0;
  return 0;
}
