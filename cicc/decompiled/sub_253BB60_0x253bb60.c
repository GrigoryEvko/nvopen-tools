// Function: sub_253BB60
// Address: 0x253bb60
//
__int64 __fastcall sub_253BB60(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // rcx
  __int64 v4; // rdi
  int v5; // eax
  int v6; // edx
  unsigned int v7; // eax
  __int64 v8; // r8
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // r8
  __int64 *v11; // r11
  int v12; // r10d
  _QWORD *v13; // rdi
  _QWORD *v14; // rsi
  _QWORD *v15; // rdi
  _QWORD *v16; // rsi
  unsigned __int64 v17; // r8
  int v18; // eax
  __int64 v19; // rcx
  int v20; // edx
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // eax
  __int64 v24; // rcx
  int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rsi
  int v28; // edi
  int v29; // r10d
  int v30; // edi
  __int64 v31; // rbp
  _QWORD v33[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( !*(_BYTE *)(a1 + 97) )
    return 0;
  v2 = a1;
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a1 + 368);
  v5 = *(_DWORD *)(v2 + 384);
  if ( !v5 )
    return 1;
  v6 = v5 - 1;
  v7 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v8 = *(_QWORD *)(v4 + 8LL * v7);
  if ( v3 != v8 )
  {
    v29 = 1;
    while ( v8 != -4096 )
    {
      v7 = v6 & (v29 + v7);
      v8 = *(_QWORD *)(v4 + 8LL * v7);
      if ( v3 == v8 )
        goto LABEL_5;
      ++v29;
    }
    return 1;
  }
LABEL_5:
  if ( *(_QWORD *)(v3 + 56) == a2 + 24 )
    return 0;
  v9 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v9 )
    return 0;
  v33[3] = v31;
  v10 = v9 - 24;
  v11 = v33;
  v12 = *(_DWORD *)(v2 + 232);
  v33[0] = v10;
  if ( v12 )
    goto LABEL_14;
LABEL_8:
  v13 = *(_QWORD **)(v2 + 248);
  v14 = &v13[*(unsigned int *)(v2 + 256)];
  if ( v14 != sub_2537F00(v13, (__int64)v14, v11) )
    return 1;
  while ( 1 )
  {
    if ( *(_DWORD *)(v2 + 120) )
    {
      v23 = *(_DWORD *)(v2 + 128);
      v24 = *(_QWORD *)(v2 + 112);
      if ( v23 )
      {
        v25 = v23 - 1;
        v26 = (v23 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v27 = *(_QWORD *)(v24 + 8LL * v26);
        if ( v10 == v27 )
          return 1;
        v28 = 1;
        while ( v27 != -4096 )
        {
          v26 = v25 & (v28 + v26);
          v27 = *(_QWORD *)(v24 + 8LL * v26);
          if ( v10 == v27 )
            return 1;
          ++v28;
        }
      }
    }
    else
    {
      v15 = *(_QWORD **)(v2 + 136);
      v16 = &v15[*(unsigned int *)(v2 + 144)];
      if ( v16 != sub_2537F00(v15, (__int64)v16, v11) )
        return 1;
    }
    if ( *(_QWORD *)(*(_QWORD *)(v10 + 40) + 56LL) == v10 + 24 )
      return 0;
    v17 = *(_QWORD *)(v10 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v17 )
      return 0;
    v10 = v17 - 24;
    v33[0] = v10;
    if ( !v12 )
      goto LABEL_8;
LABEL_14:
    v18 = *(_DWORD *)(v2 + 240);
    v19 = *(_QWORD *)(v2 + 224);
    if ( v18 )
    {
      v20 = v18 - 1;
      v21 = (v18 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v22 = *(_QWORD *)(v19 + 8LL * v21);
      if ( v10 == v22 )
        return 1;
      v30 = 1;
      while ( v22 != -4096 )
      {
        v21 = v20 & (v30 + v21);
        v22 = *(_QWORD *)(v19 + 8LL * v21);
        if ( v10 == v22 )
          return 1;
        ++v30;
      }
    }
  }
}
