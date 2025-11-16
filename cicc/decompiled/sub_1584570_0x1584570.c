// Function: sub_1584570
// Address: 0x1584570
//
__int64 __fastcall sub_1584570(_BYTE *a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r14
  unsigned int v4; // r13d
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rax

  if ( a1[16] == 9 )
    goto LABEL_9;
  if ( (unsigned __int8)sub_1593BB0(a1) )
    return sub_15A06D0(**(_QWORD **)(*(_QWORD *)a1 + 16LL));
  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 9 )
  {
LABEL_9:
    v7 = *(_QWORD **)(*(_QWORD *)a1 + 16LL);
    return sub_1599EF0(*v7);
  }
  if ( v2 != 13 )
    return 0;
  v3 = *(_QWORD *)a1;
  v4 = *(_DWORD *)(a2 + 32);
  v5 = *(unsigned int *)(*(_QWORD *)a1 + 32LL);
  if ( v4 > 0x40 )
  {
    if ( v4 - (unsigned int)sub_16A57B0(a2 + 24) > 0x40 )
      goto LABEL_7;
    v6 = **(_QWORD **)(a2 + 24);
    if ( v5 <= v6 )
      goto LABEL_7;
  }
  else
  {
    v6 = *(_QWORD *)(a2 + 24);
    if ( v5 <= v6 )
    {
LABEL_7:
      v7 = *(_QWORD **)(v3 + 16);
      return sub_1599EF0(*v7);
    }
  }
  return sub_15A0A60(a1, v6);
}
