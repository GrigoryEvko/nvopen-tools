// Function: sub_13D63A0
// Address: 0x13d63a0
//
bool __fastcall sub_13D63A0(__int64 a1, __int64 a2)
{
  char v3; // al
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rdx
  __int64 v8; // rdi
  int v9; // eax
  int v10; // eax
  __int64 v11; // r12
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __int64 v14; // rax
  char v15; // dl
  __int64 v16; // rdx
  __int64 v17; // rdi
  int v18; // eax
  int v19; // eax
  unsigned int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  int v24; // eax
  int v25; // eax
  _QWORD *v26; // rdx
  __int64 v27; // rdx

  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 == 42 )
  {
    v5 = *(_QWORD *)(a2 - 48);
    v6 = *(_BYTE *)(v5 + 16);
    if ( v6 == 37 )
    {
      v22 = *(_QWORD *)(v5 - 48);
      if ( !v22 )
        return 0;
      **(_QWORD **)a1 = v22;
      v23 = *(_QWORD *)(v5 - 24);
      v24 = *(unsigned __int8 *)(v23 + 16);
      if ( (unsigned __int8)v24 > 0x17u )
      {
        v25 = v24 - 24;
      }
      else
      {
        if ( (_BYTE)v24 != 5 )
          return 0;
        v25 = *(unsigned __int16 *)(v23 + 18);
      }
      if ( v25 != 45 )
        return 0;
      v26 = (*(_BYTE *)(v23 + 23) & 0x40) != 0
          ? *(_QWORD **)(v23 - 8)
          : (_QWORD *)(v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF));
      if ( *v26 != *(_QWORD *)(a1 + 8) )
        return 0;
    }
    else
    {
      if ( v6 != 5 )
        return 0;
      if ( *(_WORD *)(v5 + 18) != 13 )
        return 0;
      v7 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
      if ( !v7 )
        return 0;
      **(_QWORD **)a1 = v7;
      v8 = *(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)));
      v9 = *(unsigned __int8 *)(v8 + 16);
      if ( (unsigned __int8)v9 > 0x17u )
      {
        v10 = v9 - 24;
      }
      else
      {
        if ( (_BYTE)v9 != 5 )
          return 0;
        v10 = *(unsigned __int16 *)(v8 + 18);
      }
      if ( v10 != 45 || *(_QWORD *)sub_13CF970(v8) != *(_QWORD *)(a1 + 8) )
        return 0;
    }
    v11 = *(_QWORD *)(a2 - 24);
    if ( v11 )
    {
      v12 = *(_BYTE *)(v11 + 16);
      if ( v12 == 13 )
        goto LABEL_31;
      if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 16 || v12 > 0x10u )
        return 0;
      goto LABEL_18;
    }
LABEL_52:
    BUG();
  }
  if ( v3 != 5 || *(_WORD *)(a2 + 18) != 18 )
    return 0;
  v14 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v15 = *(_BYTE *)(v14 + 16);
  if ( v15 == 37 )
  {
    v27 = *(_QWORD *)(v14 - 48);
    if ( !v27 )
      return 0;
    **(_QWORD **)a1 = v27;
    v17 = *(_QWORD *)(v14 - 24);
    v18 = *(unsigned __int8 *)(v17 + 16);
    if ( (unsigned __int8)v18 > 0x17u )
      goto LABEL_26;
  }
  else
  {
    if ( v15 != 5 )
      return 0;
    if ( *(_WORD *)(v14 + 18) != 13 )
      return 0;
    v16 = *(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
    if ( !v16 )
      return 0;
    **(_QWORD **)a1 = v16;
    v17 = *(_QWORD *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
    v18 = *(unsigned __int8 *)(v17 + 16);
    if ( (unsigned __int8)v18 > 0x17u )
    {
LABEL_26:
      v19 = v18 - 24;
      goto LABEL_27;
    }
  }
  if ( (_BYTE)v18 != 5 )
    return 0;
  v19 = *(unsigned __int16 *)(v17 + 18);
LABEL_27:
  if ( v19 != 45 || *(_QWORD *)sub_13CF970(v17) != *(_QWORD *)(a1 + 8) )
    return 0;
  v11 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( !v11 )
    goto LABEL_52;
  if ( *(_BYTE *)(v11 + 16) == 13 )
    goto LABEL_31;
  if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 16 )
    return 0;
LABEL_18:
  v13 = sub_15A1020(v11);
  v11 = v13;
  if ( !v13 || *(_BYTE *)(v13 + 16) != 13 )
    return 0;
LABEL_31:
  v20 = *(_DWORD *)(v11 + 32);
  if ( v20 > 0x40 )
  {
    if ( v20 - (unsigned int)sub_16A57B0(v11 + 24) <= 0x40 )
    {
      v21 = **(_QWORD **)(v11 + 24);
      return *(_QWORD *)(a1 + 16) == v21;
    }
    return 0;
  }
  v21 = *(_QWORD *)(v11 + 24);
  return *(_QWORD *)(a1 + 16) == v21;
}
