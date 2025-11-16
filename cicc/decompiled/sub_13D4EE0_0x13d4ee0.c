// Function: sub_13D4EE0
// Address: 0x13d4ee0
//
__int64 __fastcall sub_13D4EE0(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  char v6; // al
  char v7; // dl
  __int64 v8; // r13
  __int64 v9; // rsi
  _QWORD *v11; // r15
  int v12; // eax
  int v13; // esi
  _QWORD *v14; // r14
  __int64 v15; // rcx
  __int64 v16; // rax
  char v17; // dl
  char v18; // al
  __int64 v19; // rax
  _QWORD *v20; // [rsp+0h] [rbp-40h]
  _QWORD *v21; // [rsp+8h] [rbp-38h]

  v4 = a1;
  v5 = a2;
  v6 = *(_BYTE *)(a1 + 16);
  v7 = *(_BYTE *)(a2 + 16);
  if ( (unsigned __int8)(v6 - 60) <= 0xCu )
  {
    if ( (unsigned __int8)(v7 - 60) > 0xCu )
      return 0;
    if ( v6 != v7 )
      return 0;
    v15 = *(_QWORD *)(a1 - 24);
    v5 = *(_QWORD *)(a2 - 24);
    if ( *(_QWORD *)v5 != *(_QWORD *)v15 )
      return 0;
    v7 = *(_BYTE *)(v5 + 16);
    v6 = *(_BYTE *)(v15 + 16);
    v8 = a1;
    v4 = *(_QWORD *)(a1 - 24);
  }
  else
  {
    v8 = 0;
  }
  if ( v6 != 75 )
  {
    if ( v6 != 76 || v7 != 76 )
      return 0;
    goto LABEL_9;
  }
  if ( v7 != 75 )
    return 0;
  if ( a3 )
  {
    v9 = sub_13CE210(v4, (_QWORD *)v5, 1);
    if ( !v9 )
    {
      v9 = sub_13CE210(v5, (_QWORD *)v4, 1);
      if ( !v9 )
      {
        v9 = sub_13CB980(v4, v5);
        if ( !v9 )
        {
          v9 = sub_13CB980(v5, v4);
          if ( !v9
            && (*(_QWORD *)(v4 - 48) != *(_QWORD *)(v5 - 48) || (v9 = (__int64)sub_13CD2E0((_QWORD *)v4, v5, 1)) == 0) )
          {
            v9 = sub_13D4A20(v4, v5, 1);
            if ( !v9 )
            {
              v9 = sub_13CEB50((__int64 *)v4, v5);
              if ( !v9 )
              {
                v19 = sub_13CEB50((__int64 *)v5, v4);
                v17 = *(_BYTE *)(v4 + 16);
                v9 = v19;
                v18 = *(_BYTE *)(v5 + 16);
                goto LABEL_54;
              }
            }
          }
        }
      }
    }
LABEL_32:
    if ( *(_BYTE *)(v4 + 16) != 76 || *(_BYTE *)(v5 + 16) != 76 )
      goto LABEL_33;
    goto LABEL_9;
  }
  v9 = sub_13CE210(v4, (_QWORD *)v5, 0);
  if ( v9 )
    goto LABEL_32;
  v9 = sub_13CE210(v5, (_QWORD *)v4, 0);
  if ( v9 )
    goto LABEL_32;
  v9 = sub_13CBE40(v4, v5);
  if ( v9 )
    goto LABEL_32;
  v9 = sub_13CBE40(v5, v4);
  if ( v9 )
    goto LABEL_32;
  if ( *(_QWORD *)(v4 - 48) == *(_QWORD *)(v5 - 48) )
  {
    v9 = (__int64)sub_13CD2E0((_QWORD *)v4, v5, 0);
    if ( v9 )
      goto LABEL_32;
  }
  v9 = sub_13D4A20(v4, v5, 0);
  if ( v9 )
    goto LABEL_32;
  v9 = sub_13CF520((__int64 *)v4, v5);
  if ( v9 )
    goto LABEL_32;
  v16 = sub_13CF520((__int64 *)v5, v4);
  v17 = *(_BYTE *)(v4 + 16);
  v9 = v16;
  v18 = *(_BYTE *)(v5 + 16);
LABEL_54:
  if ( v17 != 76 || v18 != 76 )
  {
    if ( !v9 )
      return 0;
    goto LABEL_33;
  }
LABEL_9:
  v11 = *(_QWORD **)(v5 - 48);
  if ( **(_QWORD **)(v4 - 48) != *v11 )
    return 0;
  v12 = *(unsigned __int16 *)(v4 + 18);
  BYTE1(v12) &= ~0x80u;
  v13 = *(_WORD *)(v5 + 18) & 0x7FFF;
  if ( v12 == 7 && v13 == 7 )
  {
    if ( !a3 )
      return 0;
  }
  else if ( v13 != 8 || v12 != 8 || a3 )
  {
    return 0;
  }
  v14 = *(_QWORD **)(v4 - 24);
  v20 = *(_QWORD **)(v4 - 48);
  v21 = *(_QWORD **)(v5 - 24);
  if ( (unsigned __int8)sub_14AB850(v20) && (v14 == v11 || v14 == v21)
    || (unsigned __int8)sub_14AB850(v14) && (v20 == v11 || v20 == v21) )
  {
    v9 = v5;
  }
  else
  {
    if ( (!(unsigned __int8)sub_14AB850(v11) || v20 != v21 && v14 != v21)
      && (!(unsigned __int8)sub_14AB850(v21) || v20 != v11 && v14 != v11) )
    {
      return 0;
    }
    v9 = v4;
  }
LABEL_33:
  if ( !v8 )
    return v9;
  if ( *(_BYTE *)(v9 + 16) > 0x10u )
    return 0;
  return sub_15A46C0((unsigned int)*(unsigned __int8 *)(v8 + 16) - 24, v9, *(_QWORD *)v8, 0);
}
