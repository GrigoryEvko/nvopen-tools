// Function: sub_20F92F0
// Address: 0x20f92f0
//
bool __fastcall sub_20F92F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r13
  unsigned int v7; // r15d
  unsigned int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rbx
  bool v14; // cf
  __int64 v15; // rax
  unsigned int v16; // ecx
  unsigned int v18; // [rsp+4h] [rbp-3Ch]

  if ( (*(_BYTE *)(a2 + 229) & 8) != 0 )
  {
    if ( (*(_BYTE *)(a3 + 229) & 8) == 0 )
      return 0;
  }
  else if ( (*(_BYTE *)(a3 + 229) & 8) != 0 )
  {
    return 1;
  }
  v6 = *(unsigned int *)(a2 + 192);
  v7 = *(_DWORD *)(a3 + 192);
  v8 = *(_DWORD *)(a2 + 192);
  v9 = **(_QWORD **)(*(_QWORD *)a1 + 16LL);
  v10 = v9 + 272 * v6;
  if ( (*(_BYTE *)(v10 + 236) & 2) == 0 )
  {
    sub_1F01F70(v9 + 272 * v6, (_QWORD *)a2, a3, a4, a5, a6);
    v9 = **(_QWORD **)(*(_QWORD *)a1 + 16LL);
  }
  v11 = v7;
  v12 = *(unsigned int *)(v10 + 244);
  v13 = v9 + 272LL * v7;
  if ( (*(_BYTE *)(v13 + 236) & 2) != 0 )
  {
    v14 = *(_DWORD *)(v13 + 244) < (unsigned int)v12;
    if ( *(_DWORD *)(v13 + 244) > (unsigned int)v12 )
      return 1;
  }
  else
  {
    v18 = v12;
    sub_1F01F70(v9 + 272LL * v7, (_QWORD *)a2, v7, v12, a5, a6);
    v11 = v7;
    v14 = *(_DWORD *)(v13 + 244) < v18;
    if ( *(_DWORD *)(v13 + 244) > v18 )
      return 1;
  }
  if ( !v14 )
  {
    v15 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
    v16 = *(_DWORD *)(v15 + 4 * v11);
    if ( *(_DWORD *)(v15 + 4 * v6) >= v16 )
    {
      if ( *(_DWORD *)(v15 + 4 * v6) <= v16 )
        return v8 > v7;
      return 0;
    }
    return 1;
  }
  return 0;
}
