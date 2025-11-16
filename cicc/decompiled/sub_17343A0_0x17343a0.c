// Function: sub_17343A0
// Address: 0x17343a0
//
__int64 __fastcall sub_17343A0(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v5; // al
  _BYTE *v7; // rdi
  unsigned __int8 v8; // al
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // rdi
  __int64 v13; // rax

  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 37 )
  {
    v7 = *(_BYTE **)(a2 - 48);
    v8 = v7[16];
    if ( v8 == 13 )
    {
      **a1 = v7 + 24;
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 )
        return 0;
      if ( v8 > 0x10u )
        return 0;
      v10 = sub_15A1020(v7, a2, *(_QWORD *)v7, a4);
      if ( !v10 || *(_BYTE *)(v10 + 16) != 13 )
        return 0;
      **a1 = v10 + 24;
    }
    v9 = *(_QWORD *)(a2 - 24);
    if ( !v9 )
      return 0;
  }
  else
  {
    if ( v5 != 5 || *(_WORD *)(a2 + 18) != 13 )
      return 0;
    v11 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v12 = *(_BYTE **)(a2 - 24 * v11);
    if ( v12[16] == 13 )
    {
      **a1 = v12 + 24;
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 16 )
        return 0;
      v13 = sub_15A1020(v12, a2, 4 * v11, a4);
      if ( !v13 || *(_BYTE *)(v13 + 16) != 13 )
        return 0;
      **a1 = v13 + 24;
    }
    v9 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !v9 )
      return 0;
  }
  *a1[1] = v9;
  return 1;
}
