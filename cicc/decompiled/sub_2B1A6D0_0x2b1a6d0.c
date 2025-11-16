// Function: sub_2B1A6D0
// Address: 0x2b1a6d0
//
__int64 __fastcall sub_2B1A6D0(__int64 a1, _BYTE *a2)
{
  bool v2; // cc
  __int64 v3; // rcx
  __int64 v4; // r9
  int v6; // esi
  __int64 v7; // r10
  int v8; // esi
  unsigned int v9; // edx
  _BYTE *v10; // rdi
  __int64 *v12; // rdx
  _QWORD *v13; // rdi
  int v14; // r8d
  unsigned int v15; // esi
  __int64 v16; // rdi
  unsigned int v17; // edx
  __int64 v18; // rcx
  _BYTE *v19; // r8
  int v20; // eax
  _QWORD *v21; // rsi
  unsigned int v22; // r8d
  int v23; // ecx
  int v24; // r11d
  __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rsi
  unsigned int v28; // edx
  __int64 *v29; // r8
  __int64 v30; // r10
  _BYTE *v31; // [rsp+8h] [rbp-18h] BYREF

  v2 = *a2 <= 0x15u;
  v31 = a2;
  if ( v2 )
    return 0;
  v3 = *(_QWORD *)(a1 + 16);
  v4 = a1;
  if ( (*(_BYTE *)(v3 + 88) & 1) != 0 )
  {
    v7 = v3 + 96;
    v8 = 3;
  }
  else
  {
    v6 = *(_DWORD *)(v3 + 104);
    v7 = *(_QWORD *)(v3 + 96);
    if ( !v6 )
      goto LABEL_13;
    v8 = v6 - 1;
  }
  v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = *(_BYTE **)(v7 + 72LL * v9);
  if ( a2 == v10 )
    return 1;
  v14 = 1;
  while ( v10 != (_BYTE *)-4096LL )
  {
    v9 = v8 & (v14 + v9);
    v10 = *(_BYTE **)(v7 + 72LL * v9);
    if ( a2 == v10 )
      return 1;
    ++v14;
  }
LABEL_13:
  v15 = *(_DWORD *)(v3 + 1192);
  v16 = *(_QWORD *)(v3 + 1176);
  if ( v15 )
  {
    v17 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v18 = v16 + 88LL * v17;
    v19 = *(_BYTE **)v18;
    if ( a2 == *(_BYTE **)v18 )
      goto LABEL_15;
    v23 = 1;
    while ( v19 != (_BYTE *)-4096LL )
    {
      v24 = v23 + 1;
      v17 = (v15 - 1) & (v23 + v17);
      v18 = v16 + 88LL * v17;
      v19 = *(_BYTE **)v18;
      if ( a2 == *(_BYTE **)v18 )
        goto LABEL_15;
      v23 = v24;
    }
  }
  v18 = v16 + 88LL * v15;
LABEL_15:
  v20 = *(_DWORD *)(v18 + 48);
  if ( v20 != 1 )
    goto LABEL_16;
  v12 = *(__int64 **)(v4 + 24);
  if ( *(_DWORD *)(v18 + 24) )
  {
    v25 = *(_QWORD *)(v18 + 16);
    v26 = *(unsigned int *)(v18 + 32);
    v27 = *v12;
    if ( (_DWORD)v26 )
    {
      v28 = (v26 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v29 = (__int64 *)(v25 + 8LL * v28);
      v30 = *v29;
      if ( v27 != *v29 )
      {
        while ( v30 != -4096 )
        {
          v28 = (v26 - 1) & (v20 + v28);
          v29 = (__int64 *)(v25 + 8LL * v28);
          v30 = *v29;
          if ( v27 == *v29 )
            goto LABEL_26;
          ++v20;
        }
        goto LABEL_16;
      }
LABEL_26:
      if ( v29 != (__int64 *)(v25 + 8 * v26) )
        return 0;
    }
  }
  else
  {
    v13 = *(_QWORD **)(v18 + 40);
    if ( v13 + 1 != sub_2B0B4F0(v13, (__int64)(v13 + 1), v12) )
      return 0;
  }
LABEL_16:
  v21 = (_QWORD *)(*(_QWORD *)v4 + 8LL * *(_QWORD *)(v4 + 8));
  LOBYTE(v22) = v21 == sub_2B0CA10(*(_QWORD **)v4, (__int64)v21, (__int64 *)&v31);
  return v22;
}
