// Function: sub_13D1770
// Address: 0x13d1770
//
__int64 __fastcall sub_13D1770(_BYTE *a1, __int64 a2)
{
  __int64 result; // rax
  char v3; // dl
  __int64 v4; // r14
  unsigned int v5; // ebx
  unsigned __int64 v6; // r15
  _QWORD *v7; // rax

  if ( a1[16] > 0x10u )
    goto LABEL_6;
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_1584570();
  result = sub_15A1020(a1);
  if ( !result )
  {
    if ( a1[16] == 9 )
    {
LABEL_18:
      v7 = *(_QWORD **)(*(_QWORD *)a1 + 16LL);
      return sub_1599EF0(*v7);
    }
LABEL_6:
    v3 = *(_BYTE *)(a2 + 16);
    if ( v3 != 13 )
      goto LABEL_14;
    v4 = *(_QWORD *)a1;
    v5 = *(_DWORD *)(a2 + 32);
    v6 = *(unsigned int *)(*(_QWORD *)a1 + 32LL);
    if ( v5 <= 0x40 )
    {
      if ( v6 > *(_QWORD *)(a2 + 24) )
      {
LABEL_12:
        result = sub_14C48A0(a1);
        if ( result )
          return result;
        v3 = *(_BYTE *)(a2 + 16);
LABEL_14:
        result = 0;
        if ( v3 != 9 )
          return result;
        goto LABEL_18;
      }
    }
    else if ( v5 - (unsigned int)sub_16A57B0(a2 + 24) <= 0x40 && v6 > **(_QWORD **)(a2 + 24) )
    {
      goto LABEL_12;
    }
    v7 = *(_QWORD **)(v4 + 16);
    return sub_1599EF0(*v7);
  }
  return result;
}
