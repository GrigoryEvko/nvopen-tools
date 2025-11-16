// Function: sub_175DA00
// Address: 0x175da00
//
__int64 __fastcall sub_175DA00(__int64 **a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  _BYTE *v6; // r13
  unsigned __int8 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r13

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 != 48 )
  {
    if ( v4 != 5 || *(_WORD *)(a2 + 18) != 24 )
      return 0;
    v12 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v13 = 4 * v12;
    v14 = *(_BYTE **)(a2 - 24 * v12);
    if ( v14[16] == 13 )
    {
      if ( *((_DWORD *)v14 + 8) <= 0x40u )
      {
        v16 = *((_QWORD *)v14 + 3);
        if ( v16 )
        {
          v13 = v16 - 1;
          if ( (v16 & (v16 - 1)) == 0 )
            goto LABEL_20;
        }
      }
      else if ( (unsigned int)sub_16A5940((__int64)(v14 + 24)) == 1 )
      {
LABEL_20:
        **a1 = (__int64)(v14 + 24);
        goto LABEL_21;
      }
    }
    if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 16 )
      return 0;
    v17 = sub_15A1020(v14, a2, v13, a4);
    if ( !v17 )
      return 0;
    if ( *(_BYTE *)(v17 + 16) != 13 )
      return 0;
    v18 = v17 + 24;
    if ( !sub_14A9C60(v17 + 24) )
      return 0;
    **a1 = v18;
LABEL_21:
    v15 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !v15 )
      return 0;
    goto LABEL_33;
  }
  v6 = *(_BYTE **)(a2 - 48);
  v7 = v6[16];
  if ( v7 != 13 )
    goto LABEL_9;
  if ( *((_DWORD *)v6 + 8) <= 0x40u )
  {
    v8 = *((_QWORD *)v6 + 3);
    if ( !v8 )
      goto LABEL_9;
    a3 = v8 - 1;
    if ( (v8 & (v8 - 1)) != 0 )
      goto LABEL_9;
LABEL_31:
    **a1 = (__int64)(v6 + 24);
    goto LABEL_32;
  }
  if ( (unsigned int)sub_16A5940((__int64)(v6 + 24)) == 1 )
    goto LABEL_31;
LABEL_9:
  if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) != 16 )
    return 0;
  if ( v7 > 0x10u )
    return 0;
  v9 = sub_15A1020(v6, a2, a3, a4);
  if ( !v9 || *(_BYTE *)(v9 + 16) != 13 )
    return 0;
  v10 = v9 + 24;
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
  {
    if ( (unsigned int)sub_16A5940(v9 + 24) != 1 )
      return 0;
  }
  else
  {
    v11 = *(_QWORD *)(v9 + 24);
    if ( !v11 || (v11 & (v11 - 1)) != 0 )
      return 0;
  }
  **a1 = v10;
LABEL_32:
  v15 = *(_QWORD *)(a2 - 24);
  if ( !v15 )
    return 0;
LABEL_33:
  *a1[1] = v15;
  return 1;
}
