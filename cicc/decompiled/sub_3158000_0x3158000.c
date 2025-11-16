// Function: sub_3158000
// Address: 0x3158000
//
bool __fastcall sub_3158000(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // r8
  __int64 v5; // rax
  _QWORD *v6; // rdi
  _QWORD *v7; // rsi
  __int64 v8; // rax
  _QWORD *v9; // rdi
  _QWORD *v10; // rsi
  bool result; // al
  __int64 v12; // rcx
  int v13; // eax
  int v14; // edx
  unsigned int v15; // eax
  __int64 v16; // rsi
  int v17; // edi
  __int64 v18; // rsi
  int v19; // eax
  int v20; // ecx
  unsigned int v21; // edx
  __int64 v22; // rdi
  int v23; // eax
  __int64 v24[2]; // [rsp+8h] [rbp-18h] BYREF

  v24[0] = sub_B46EC0(a2, a3);
  v4 = v24[0];
  v5 = *a1;
  if ( *(_DWORD *)(*a1 + 16LL) )
  {
    v12 = *(_QWORD *)(v5 + 8);
    v13 = *(_DWORD *)(v5 + 24);
    if ( !v13 )
      return 0;
    v14 = v13 - 1;
    v15 = (v13 - 1) & ((LODWORD(v24[0]) >> 9) ^ (LODWORD(v24[0]) >> 4));
    v16 = *(_QWORD *)(v12 + 8LL * v15);
    if ( v24[0] != v16 )
    {
      v17 = 1;
      while ( v16 != -4096 )
      {
        v15 = v14 & (v17 + v15);
        v16 = *(_QWORD *)(v12 + 8LL * v15);
        if ( v24[0] == v16 )
          goto LABEL_3;
        ++v17;
      }
      return 0;
    }
  }
  else
  {
    v6 = *(_QWORD **)(v5 + 32);
    v7 = &v6[*(unsigned int *)(v5 + 40)];
    if ( v7 == sub_3157E90(v6, (__int64)v7, v24) )
      return 0;
  }
LABEL_3:
  v8 = a1[1];
  if ( !*(_DWORD *)(v8 + 16) )
  {
    v9 = *(_QWORD **)(v8 + 32);
    v10 = &v9[*(unsigned int *)(v8 + 40)];
    return v10 != sub_3157E90(v9, (__int64)v10, v24);
  }
  v18 = *(_QWORD *)(v8 + 8);
  v19 = *(_DWORD *)(v8 + 24);
  if ( !v19 )
    return 0;
  v20 = v19 - 1;
  v21 = (v19 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v22 = *(_QWORD *)(v18 + 8LL * v21);
  result = 1;
  if ( v4 != v22 )
  {
    v23 = 1;
    while ( v22 != -4096 )
    {
      v21 = v20 & (v23 + v21);
      v22 = *(_QWORD *)(v18 + 8LL * v21);
      if ( v4 == v22 )
        return 1;
      ++v23;
    }
    return 0;
  }
  return result;
}
