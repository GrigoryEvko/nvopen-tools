// Function: sub_B903E0
// Address: 0xb903e0
//
__int64 __fastcall sub_B903E0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rcx
  int v4; // r8d
  int v5; // r13d
  unsigned int i; // r15d
  __int64 *v9; // r14
  __int64 v10; // rdi
  unsigned int v11; // r15d
  __int64 v12; // r9
  __int64 v13; // rax
  _BYTE *v14; // rax
  _QWORD *v15; // rsi
  __int64 v17; // rax
  _BYTE *v18; // rax
  _QWORD *v19; // rsi
  _QWORD *v20; // rdx
  __int64 v21; // [rsp-58h] [rbp-58h]
  __int64 v22; // [rsp-58h] [rbp-58h]
  int v23; // [rsp-4Ch] [rbp-4Ch]
  int v24; // [rsp-4Ch] [rbp-4Ch]
  __int64 v25; // [rsp-48h] [rbp-48h]
  __int64 v26; // [rsp-48h] [rbp-48h]
  _QWORD *v27; // [rsp-40h] [rbp-40h]
  _QWORD *v28; // [rsp-40h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return 0;
  v4 = 1;
  v5 = v2 - 1;
  for ( i = (v2 - 1) & *(_DWORD *)(a2 + 32); ; i = v5 & v11 )
  {
    v9 = (__int64 *)(v3 + 8LL * i);
    v10 = *v9;
    if ( *v9 == -8192 )
      goto LABEL_15;
    if ( v10 == -4096 )
      return 0;
    if ( *(_DWORD *)(a2 + 32) != *(_DWORD *)(v10 + 4) )
      goto LABEL_6;
    v12 = *(_QWORD *)(a2 + 8);
    if ( v12 )
    {
      if ( (*(_BYTE *)(v10 - 16) & 2) != 0 )
        v13 = *(unsigned int *)(v10 - 24);
      else
        v13 = (*(_WORD *)(v10 - 16) >> 6) & 0xF;
      if ( v12 != v13 )
        goto LABEL_6;
      v21 = *(_QWORD *)(a2 + 8);
      v23 = v4;
      v25 = v3;
      v27 = *(_QWORD **)a2;
      v14 = sub_AF15A0((_BYTE *)(v10 - 16));
      v15 = v27;
      v4 = v23;
      v3 = v25;
      while ( *v15 == *(_QWORD *)v14 )
      {
        ++v15;
        v14 += 8;
        if ( &v27[v21] == v15 )
          goto LABEL_26;
      }
      goto LABEL_14;
    }
    if ( (*(_BYTE *)(v10 - 16) & 2) != 0 )
      v17 = *(unsigned int *)(v10 - 24);
    else
      v17 = (*(_WORD *)(v10 - 16) >> 6) & 0xF;
    if ( v17 != *(_QWORD *)(a2 + 24) )
      goto LABEL_6;
    v22 = *(_QWORD *)(a2 + 24);
    v24 = v4;
    v26 = v3;
    v28 = *(_QWORD **)(a2 + 16);
    v18 = sub_AF15A0((_BYTE *)(v10 - 16));
    v19 = v28;
    v3 = v26;
    v4 = v24;
    v20 = &v28[v22];
    if ( v20 == v28 )
      break;
    while ( *v19 == *(_QWORD *)v18 )
    {
      ++v19;
      v18 += 8;
      if ( v20 == v19 )
        goto LABEL_26;
    }
LABEL_14:
    v10 = *v9;
LABEL_15:
    if ( v10 == -4096 )
      return 0;
LABEL_6:
    v11 = v4 + i;
    ++v4;
  }
LABEL_26:
  if ( v9 == (__int64 *)(*(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 24)) )
    return 0;
  return *v9;
}
