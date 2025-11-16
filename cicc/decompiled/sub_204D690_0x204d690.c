// Function: sub_204D690
// Address: 0x204d690
//
char __fastcall sub_204D690(char a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  unsigned __int64 v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // eax
  __int64 v13; // rcx
  unsigned int v14; // eax
  char v15; // dl
  char v16; // dl
  __int64 v17; // rsi

  if ( !a2 )
    return 1;
  if ( *(_BYTE *)(a2 + 16) != 18 )
  {
    v6 = *(_QWORD *)a2;
    if ( a1 )
    {
      if ( *(_BYTE *)(v6 + 8) != 15 )
        sub_16BD130("Indirect operand for inline asm not a pointer!", 1u);
      v6 = *(_QWORD *)(v6 + 24);
      v9 = *(unsigned __int8 *)(v6 + 8);
      if ( (_BYTE)v9 != 13 )
        goto LABEL_13;
    }
    else
    {
      v9 = *(unsigned __int8 *)(v6 + 8);
      if ( (_BYTE)v9 != 13 )
        goto LABEL_13;
    }
    if ( *(_DWORD *)(v6 + 12) != 1 )
      goto LABEL_6;
    v6 = **(_QWORD **)(v6 + 16);
    v9 = *(unsigned __int8 *)(v6 + 8);
LABEL_13:
    if ( (unsigned __int8)v9 > 0x10u )
    {
LABEL_14:
      if ( (unsigned int)(v9 - 13) > 1 )
        return sub_204D4D0(a4, a5, v6);
      goto LABEL_15;
    }
    v13 = 100990;
    if ( _bittest64(&v13, v9) )
      return sub_204D4D0(a4, a5, v6);
    if ( (_BYTE)v9 == 16 )
    {
LABEL_15:
      if ( !sub_16435F0(v6, 0) )
        return sub_204D4D0(a4, a5, v6);
LABEL_7:
      v11 = sub_127FA20(a5, v6);
      if ( v11 > 0x40 )
      {
        if ( v11 != 128 )
          return sub_204D4D0(a4, a5, v6);
      }
      else if ( v11 > 7 )
      {
        v17 = 0x100000001000101LL;
        if ( !_bittest64(&v17, v11 - 8) )
          return sub_204D4D0(a4, a5, v6);
      }
      else if ( v11 != 1 )
      {
        return sub_204D4D0(a4, a5, v6);
      }
      v6 = sub_1644900(a3, v11);
      return sub_204D4D0(a4, a5, v6);
    }
LABEL_6:
    v10 = 35454;
    if ( _bittest64(&v10, v9) )
      goto LABEL_7;
    goto LABEL_14;
  }
  v14 = 8 * sub_15A9520(a5, 0);
  if ( v14 == 32 )
    return 5;
  if ( v14 <= 0x20 )
  {
    v15 = 3;
    if ( v14 != 8 )
      return 4 * (v14 == 16);
    return v15;
  }
  v15 = 6;
  if ( v14 == 64 )
    return v15;
  v16 = 0;
  if ( v14 == 128 )
    return 7;
  return v16;
}
