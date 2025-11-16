// Function: sub_277B370
// Address: 0x277b370
//
char __fastcall sub_277B370(__int64 a1, int a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  int v7; // edx
  __int64 v9; // r8
  int v10; // edx
  unsigned int v11; // esi
  __int64 *v12; // rax
  __int64 v13; // r9
  __int64 v14; // r13
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // rsi
  _BYTE *v18; // rax
  __int64 *v19; // rdx
  bool v20; // zf
  __int64 *v21; // rax
  __int64 v22; // rsi
  int v24; // eax
  _QWORD *v25; // rax
  __int64 v26; // rax
  int v27; // r10d
  int v28; // eax
  int v29; // r9d

  if ( a2 == a3 )
    return 1;
  v6 = *(_QWORD *)(a1 + 104);
  if ( !v6 )
    return 0;
  v7 = *(_DWORD *)(v6 + 56);
  v9 = *(_QWORD *)(v6 + 40);
  if ( !v7 )
    return 1;
  v10 = v7 - 1;
  v11 = v10 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( a4 != *v12 )
  {
    v24 = 1;
    while ( v13 != -4096 )
    {
      v27 = v24 + 1;
      v11 = v10 & (v24 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( a4 == *v12 )
        goto LABEL_5;
      v24 = v27;
    }
    return 1;
  }
LABEL_5:
  v14 = v12[1];
  if ( !v14 )
    return 1;
  v15 = v10 & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
  v16 = (__int64 *)(v9 + 16LL * v15);
  v17 = *v16;
  if ( a5 != *v16 )
  {
    v28 = 1;
    while ( v17 != -4096 )
    {
      v29 = v28 + 1;
      v15 = v10 & (v28 + v15);
      v16 = (__int64 *)(v9 + 16LL * v15);
      v17 = *v16;
      if ( a5 == *v16 )
        goto LABEL_7;
      v28 = v29;
    }
    return 1;
  }
LABEL_7:
  v18 = (_BYTE *)v16[1];
  if ( !v18 )
    return 1;
  if ( *(_DWORD *)(a1 + 740) < (unsigned int)qword_4FFB268 )
  {
    v25 = sub_103E0E0((_QWORD *)v6);
    v26 = sub_277B150(v25, a5);
    ++*(_DWORD *)(a1 + 740);
    v6 = *(_QWORD *)(a1 + 104);
    v22 = v26;
  }
  else
  {
    v19 = (__int64 *)(v18 - 64);
    v20 = *v18 == 26;
    v21 = (__int64 *)(v18 - 32);
    if ( !v20 )
      v21 = v19;
    v22 = *v21;
  }
  return sub_1041420(v6, v22, v14);
}
