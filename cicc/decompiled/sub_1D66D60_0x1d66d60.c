// Function: sub_1D66D60
// Address: 0x1d66d60
//
bool __fastcall sub_1D66D60(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char v5; // al
  __int64 v6; // rdi
  __int64 v7; // rax
  int v8; // eax
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned int *v12; // r12
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  int v18; // eax
  int v19; // eax
  _QWORD *v20; // rax
  unsigned int v21; // r13d
  __int64 v22; // rax

  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 != 47 )
  {
    if ( v5 != 5 )
      return 0;
    if ( *(_WORD *)(a2 + 18) != 23 )
      return 0;
    v16 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v17 = *(_QWORD *)(v16 + 8);
    if ( !v17 || *(_QWORD *)(v17 + 8) )
      return 0;
    v18 = *(unsigned __int8 *)(v16 + 16);
    if ( (unsigned __int8)v18 > 0x17u )
    {
      v19 = v18 - 24;
    }
    else
    {
      if ( (_BYTE)v18 != 5 )
        return 0;
      v19 = *(unsigned __int16 *)(v16 + 18);
    }
    if ( v19 != 37 )
      return 0;
    v20 = (_QWORD *)sub_13CF970(v16);
    if ( !*v20 )
      return 0;
    **(_QWORD **)a1 = *v20;
    v14 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v12 = *(unsigned int **)(a2 + 24 * (1 - v14));
    if ( !v12 )
      BUG();
    if ( *((_BYTE *)v12 + 16) == 13 )
      goto LABEL_32;
    if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 16 )
      return 0;
    goto LABEL_20;
  }
  v6 = *(_QWORD *)(a2 - 48);
  v7 = *(_QWORD *)(v6 + 8);
  if ( !v7 || *(_QWORD *)(v7 + 8) )
    return 0;
  v8 = *(unsigned __int8 *)(v6 + 16);
  if ( (unsigned __int8)v8 > 0x17u )
  {
    v9 = v8 - 24;
  }
  else
  {
    if ( (_BYTE)v8 != 5 )
      return 0;
    v9 = *(unsigned __int16 *)(v6 + 18);
  }
  if ( v9 != 37 )
    return 0;
  v10 = *(_QWORD *)sub_13CF970(v6);
  if ( !v10 )
    return 0;
  **(_QWORD **)a1 = v10;
  v12 = *(unsigned int **)(a2 - 24);
  if ( !v12 )
    BUG();
  v13 = *((_BYTE *)v12 + 16);
  if ( v13 != 13 )
  {
    v14 = *(_QWORD *)v12;
    if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 16 || v13 > 0x10u )
      return 0;
LABEL_20:
    v15 = sub_15A1020(v12, a2, v14, v11);
    v12 = (unsigned int *)v15;
    if ( !v15 || *(_BYTE *)(v15 + 16) != 13 )
      return 0;
  }
LABEL_32:
  v21 = v12[8];
  if ( v21 <= 0x40 )
  {
    v22 = *((_QWORD *)v12 + 3);
    return *(_QWORD *)(a1 + 8) == v22;
  }
  if ( v21 - (unsigned int)sub_16A57B0((__int64)(v12 + 6)) <= 0x40 )
  {
    v22 = **((_QWORD **)v12 + 3);
    return *(_QWORD *)(a1 + 8) == v22;
  }
  return 0;
}
