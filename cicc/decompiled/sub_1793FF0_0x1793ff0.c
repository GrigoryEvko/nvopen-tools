// Function: sub_1793FF0
// Address: 0x1793ff0
//
__int64 __fastcall sub_1793FF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r12
  char v5; // al
  _BYTE *v7; // r13
  unsigned __int8 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // r12
  _BYTE *v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax

  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 50 )
  {
    v7 = *(_BYTE **)(a2 - 24);
    v8 = v7[16];
    if ( v8 == 13 )
    {
      v4 = v7 + 24;
      if ( *((_DWORD *)v7 + 8) > 0x40u )
      {
        if ( (unsigned int)sub_16A5940((__int64)(v7 + 24)) == 1 )
          goto LABEL_29;
      }
      else
      {
        v9 = *((_QWORD *)v7 + 3);
        if ( v9 )
        {
          a3 = v9 - 1;
          if ( (v9 & (v9 - 1)) == 0 )
            goto LABEL_29;
        }
      }
    }
    LOBYTE(v4) = v8 <= 0x10u && *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16;
    if ( !(_BYTE)v4 )
      goto LABEL_4;
    v10 = sub_15A1020(v7, a2, a3, a4);
    if ( !v10 || *(_BYTE *)(v10 + 16) != 13 )
      goto LABEL_4;
    v11 = v10 + 24;
    if ( *(_DWORD *)(v10 + 32) > 0x40u )
    {
      if ( (unsigned int)sub_16A5940(v10 + 24) != 1 )
        goto LABEL_4;
    }
    else
    {
      v12 = *(_QWORD *)(v10 + 24);
      if ( !v12 || (v12 & (v12 - 1)) != 0 )
        goto LABEL_4;
    }
    **(_QWORD **)(a1 + 8) = v11;
    return (unsigned int)v4;
  }
  if ( v5 != 5 || *(_WORD *)(a2 + 18) != 26 )
    goto LABEL_4;
  v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v14 = *(_BYTE **)(a2 + 24 * (1 - v13));
  if ( v14[16] == 13 )
  {
    v15 = v14 + 24;
    if ( *((_DWORD *)v14 + 8) <= 0x40u )
    {
      v16 = *((_QWORD *)v14 + 3);
      if ( !v16 )
        goto LABEL_21;
      v13 = v16 - 1;
      if ( (v16 & (v16 - 1)) != 0 )
        goto LABEL_21;
LABEL_31:
      LODWORD(v4) = 1;
      **(_QWORD **)(a1 + 8) = v15;
      return (unsigned int)v4;
    }
    if ( (unsigned int)sub_16A5940((__int64)(v14 + 24)) == 1 )
      goto LABEL_31;
  }
LABEL_21:
  if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 16 )
    goto LABEL_4;
  v17 = sub_15A1020(v14, a2, v13, a4);
  if ( !v17 || *(_BYTE *)(v17 + 16) != 13 )
    goto LABEL_4;
  v4 = (_BYTE *)(v17 + 24);
  if ( *(_DWORD *)(v17 + 32) <= 0x40u )
  {
    v18 = *(_QWORD *)(v17 + 24);
    if ( !v18 || (v18 & (v18 - 1)) != 0 )
      goto LABEL_4;
LABEL_29:
    **(_QWORD **)(a1 + 8) = v4;
    LODWORD(v4) = 1;
    return (unsigned int)v4;
  }
  if ( (unsigned int)sub_16A5940(v17 + 24) == 1 )
    goto LABEL_29;
LABEL_4:
  LODWORD(v4) = 0;
  return (unsigned int)v4;
}
