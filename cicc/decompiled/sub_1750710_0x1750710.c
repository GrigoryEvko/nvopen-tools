// Function: sub_1750710
// Address: 0x1750710
//
bool __fastcall sub_1750710(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  unsigned int *v6; // r13
  unsigned __int8 v7; // al
  unsigned int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // rax

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 37 )
  {
    v6 = *(unsigned int **)(a2 - 48);
    if ( v6 )
    {
      v7 = *((_BYTE *)v6 + 16);
      if ( v7 != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) != 16 )
          return 0;
        if ( v7 > 0x10u )
          return 0;
        v10 = sub_15A1020(*(_BYTE **)(a2 - 48), a2, *(_QWORD *)v6, a4);
        v6 = (unsigned int *)v10;
        if ( !v10 || *(_BYTE *)(v10 + 16) != 13 )
          return 0;
      }
      v8 = v6[8];
      if ( v8 <= 0x40 )
      {
        v9 = *((_QWORD *)v6 + 3);
      }
      else
      {
        if ( v8 - (unsigned int)sub_16A57B0((__int64)(v6 + 6)) > 0x40 )
          return 0;
        v9 = **((_QWORD **)v6 + 3);
      }
      if ( *a1 == v9 )
        return a1[1] == *(_QWORD *)(a2 - 24);
      return 0;
    }
LABEL_30:
    BUG();
  }
  if ( v4 != 5 || *(_WORD *)(a2 + 18) != 13 )
    return 0;
  v11 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v12 = *(_QWORD *)(a2 - 24 * v11);
  if ( !v12 )
    goto LABEL_30;
  if ( *(_BYTE *)(v12 + 16) != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 16 )
      return 0;
    v15 = sub_15A1020((_BYTE *)v12, a2, 4 * v11, a4);
    v12 = v15;
    if ( !v15 || *(_BYTE *)(v15 + 16) != 13 )
      return 0;
  }
  v13 = *(_DWORD *)(v12 + 32);
  if ( v13 > 0x40 )
  {
    if ( v13 - (unsigned int)sub_16A57B0(v12 + 24) <= 0x40 )
    {
      v14 = **(_QWORD **)(v12 + 24);
      goto LABEL_22;
    }
    return 0;
  }
  v14 = *(_QWORD *)(v12 + 24);
LABEL_22:
  if ( *a1 != v14 )
    return 0;
  return *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) == a1[1];
}
