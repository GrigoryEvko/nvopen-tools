// Function: sub_35E86F0
// Address: 0x35e86f0
//
__int64 __fastcall sub_35E86F0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  int v6; // r14d
  __int64 v7; // r8
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // r10
  int v13; // eax
  int v14; // ecx
  int v15; // eax
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 v19; // rdi
  int v20; // r10d
  _QWORD *v21; // r9
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  _QWORD *v25; // r8
  unsigned int v26; // r13d
  int v27; // r9d
  __int64 v28; // rsi

  v4 = a1 + 208;
  v5 = *(_DWORD *)(a1 + 232);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 208);
    goto LABEL_18;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 216);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
    return v10[1];
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v5 - 1) & (v6 + v9);
    v10 = (_QWORD *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      return v10[1];
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 224);
  ++*(_QWORD *)(a1 + 208);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v5 )
  {
LABEL_18:
    sub_2E48800(v4, 2 * v5);
    v15 = *(_DWORD *)(a1 + 232);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 216);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 224) + 1;
      v8 = (_QWORD *)(v17 + 16LL * v18);
      v19 = *v8;
      if ( *v8 != a2 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -4096 )
        {
          if ( !v21 && v19 == -8192 )
            v21 = v8;
          v18 = v16 & (v20 + v18);
          v8 = (_QWORD *)(v17 + 16LL * v18);
          v19 = *v8;
          if ( *v8 == a2 )
            goto LABEL_14;
          ++v20;
        }
        if ( v21 )
          v8 = v21;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v5 - *(_DWORD *)(a1 + 228) - v14 <= v5 >> 3 )
  {
    sub_2E48800(v4, v5);
    v22 = *(_DWORD *)(a1 + 232);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 216);
      v25 = 0;
      v26 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = 1;
      v14 = *(_DWORD *)(a1 + 224) + 1;
      v8 = (_QWORD *)(v24 + 16LL * v26);
      v28 = *v8;
      if ( *v8 != a2 )
      {
        while ( v28 != -4096 )
        {
          if ( !v25 && v28 == -8192 )
            v25 = v8;
          v26 = v23 & (v27 + v26);
          v8 = (_QWORD *)(v24 + 16LL * v26);
          v28 = *v8;
          if ( *v8 == a2 )
            goto LABEL_14;
          ++v27;
        }
        if ( v25 )
          v8 = v25;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(a1 + 224);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 224) = v14;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 228);
  *v8 = a2;
  v8[1] = 0;
  return 0;
}
