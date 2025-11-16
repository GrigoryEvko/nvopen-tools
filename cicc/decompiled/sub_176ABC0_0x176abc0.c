// Function: sub_176ABC0
// Address: 0x176abc0
//
bool __fastcall sub_176ABC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 v6; // rax
  unsigned int *v7; // r12
  unsigned __int8 v8; // al
  unsigned int v9; // r13d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 49 )
  {
    v6 = *(_QWORD *)(a2 - 48);
    if ( !v6 )
      return 0;
    **(_QWORD **)a1 = v6;
    v7 = *(unsigned int **)(a2 - 24);
    if ( v7 )
    {
      v8 = *((_BYTE *)v7 + 16);
      if ( v8 == 13 )
        goto LABEL_8;
      v11 = *(_QWORD *)v7;
      if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 || v8 > 0x10u )
        return 0;
      goto LABEL_20;
    }
LABEL_23:
    BUG();
  }
  if ( v4 != 5 )
    return 0;
  if ( *(_WORD *)(a2 + 18) != 25 )
    return 0;
  v12 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( !v12 )
    return 0;
  **(_QWORD **)a1 = v12;
  v11 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v7 = *(unsigned int **)(a2 + 24 * (1 - v11));
  if ( !v7 )
    goto LABEL_23;
  if ( *((_BYTE *)v7 + 16) == 13 )
    goto LABEL_8;
  if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 )
    return 0;
LABEL_20:
  v13 = sub_15A1020(v7, a2, v11, a4);
  v7 = (unsigned int *)v13;
  if ( !v13 || *(_BYTE *)(v13 + 16) != 13 )
    return 0;
LABEL_8:
  v9 = v7[8];
  if ( v9 <= 0x40 )
  {
    v10 = *((_QWORD *)v7 + 3);
    return *(_QWORD *)(a1 + 8) == v10;
  }
  if ( v9 - (unsigned int)sub_16A57B0((__int64)(v7 + 6)) > 0x40 )
    return 0;
  v10 = **((_QWORD **)v7 + 3);
  return *(_QWORD *)(a1 + 8) == v10;
}
