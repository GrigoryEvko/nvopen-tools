// Function: sub_89B9E0
// Address: 0x89b9e0
//
__int64 __fastcall sub_89B9E0(__int64 a1, __int64 a2, char a3, unsigned int a4)
{
  unsigned int v4; // r12d
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r8
  int v14; // eax

  v4 = a3 & 1;
  if ( (a3 & 1) != 0 )
    return a1 == a2;
  if ( (*(_BYTE *)(a1 + 160) & 2) != 0 && (*(_BYTE *)(a2 + 160) & 2) != 0 )
  {
    v6 = *(_QWORD *)(a1 + 104);
    v7 = *(_QWORD *)(a2 + 104);
    if ( !strcmp(*(const char **)(v6 + 8), *(const char **)(v7 + 8)) )
    {
      v10 = *(_QWORD *)(*(_QWORD *)(v6 + 40) + 32LL);
      v11 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 32LL);
      if ( v10 == v11 || (unsigned int)sub_8D97D0(v10, v11, 0, v8, v9) )
        return 1;
    }
  }
  else
  {
    if ( (*(_BYTE *)(a1 + 266) & 1) == 0 || (*(_BYTE *)(a2 + 266) & 1) == 0 )
      return a1 == a2;
    v12 = *(_QWORD *)(a2 + 104);
    v13 = *(_QWORD *)(a1 + 104);
    if ( *(_DWORD *)(v13 + 128) == *(_DWORD *)(v12 + 128) || (a3 & 2) != 0 )
    {
      v14 = *(_DWORD *)(v12 + 132);
      if ( *(_DWORD *)(v13 + 132) == 0 || *(_DWORD *)(v13 + 132) == v14 || !v14 )
        return (unsigned int)sub_89B3C0(**(_QWORD **)(a1 + 32), **(_QWORD **)(a2 + 32), 0, a4, 0, 8u) != 0;
    }
  }
  return v4;
}
