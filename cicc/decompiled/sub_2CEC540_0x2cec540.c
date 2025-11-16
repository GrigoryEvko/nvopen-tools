// Function: sub_2CEC540
// Address: 0x2cec540
//
__int64 __fastcall sub_2CEC540(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 *v5; // rax
  __int64 v7; // rbx
  unsigned int v8; // esi
  __int64 v9; // r8
  int v10; // r10d
  __int64 *v11; // rdx
  unsigned int v12; // edi
  __int64 *v13; // rax
  __int64 v14; // rcx
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
  unsigned int v31[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(__int64 **)a1;
  v31[0] = 0;
  if ( (unsigned __int8)sub_2CE0930(*(_QWORD **)(a1 + 8), *v5, a2, v31) )
  {
    *a3 = 1;
    return v31[0];
  }
  *a3 = 0;
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_DWORD *)(v7 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)v7;
    goto LABEL_20;
  }
  v9 = *(_QWORD *)(v7 + 8);
  v10 = 1;
  v11 = 0;
  v12 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v9 + 16LL * v12);
  v14 = *v13;
  if ( *v13 == a2 )
    return *((unsigned int *)v13 + 2);
  while ( v14 != -4096 )
  {
    if ( !v11 && v14 == -8192 )
      v11 = v13;
    v12 = (v8 - 1) & (v10 + v12);
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( *v13 == a2 )
      return *((unsigned int *)v13 + 2);
    ++v10;
  }
  if ( !v11 )
    v11 = v13;
  v15 = *(_DWORD *)(v7 + 16);
  ++*(_QWORD *)v7;
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v8 )
  {
LABEL_20:
    sub_D39D40(v7, 2 * v8);
    v17 = *(_DWORD *)(v7 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v7 + 8);
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(v7 + 16) + 1;
      v11 = (__int64 *)(v19 + 16LL * v20);
      v21 = *v11;
      if ( *v11 != a2 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4096 )
        {
          if ( !v23 && v21 == -8192 )
            v23 = v11;
          v20 = v18 & (v22 + v20);
          v11 = (__int64 *)(v19 + 16LL * v20);
          v21 = *v11;
          if ( *v11 == a2 )
            goto LABEL_16;
          ++v22;
        }
        if ( v23 )
          v11 = v23;
      }
      goto LABEL_16;
    }
    goto LABEL_43;
  }
  if ( v8 - *(_DWORD *)(v7 + 20) - v16 <= v8 >> 3 )
  {
    sub_D39D40(v7, v8);
    v24 = *(_DWORD *)(v7 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(v7 + 8);
      v27 = 1;
      v28 = v25 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v29 = 0;
      v16 = *(_DWORD *)(v7 + 16) + 1;
      v11 = (__int64 *)(v26 + 16LL * v28);
      v30 = *v11;
      if ( *v11 != a2 )
      {
        while ( v30 != -4096 )
        {
          if ( !v29 && v30 == -8192 )
            v29 = v11;
          v28 = v25 & (v27 + v28);
          v11 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v11;
          if ( *v11 == a2 )
            goto LABEL_16;
          ++v27;
        }
        if ( v29 )
          v11 = v29;
      }
      goto LABEL_16;
    }
LABEL_43:
    ++*(_DWORD *)(v7 + 16);
    BUG();
  }
LABEL_16:
  *(_DWORD *)(v7 + 16) = v16;
  if ( *v11 != -4096 )
    --*(_DWORD *)(v7 + 20);
  *v11 = a2;
  *((_DWORD *)v11 + 2) = 0;
  return 0;
}
