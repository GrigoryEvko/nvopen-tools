// Function: sub_5CF930
// Address: 0x5cf930
//
__int64 __fastcall sub_5CF930(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  char v5; // dl
  __int64 i; // rax
  __int64 v7; // rax
  char v9; // al
  __int64 v10; // rdi

  if ( a3 == 11 )
  {
    v9 = *(_BYTE *)(a2 + 174);
    if ( v9 == 2
      || (v10 = *(_QWORD *)(*(_QWORD *)(a2 + 152) + 160LL)) != 0
      && v9 != 1
      && (unsigned int)sub_8D2600(v10)
      && (*(_BYTE *)(a2 + 195) & 9) != 1
      && !(unsigned int)sub_5F26C0() )
    {
      sub_684B30(2811, a1 + 56);
      *(_BYTE *)(a1 + 8) = 0;
      return a2;
    }
    *(_BYTE *)(a2 + 196) |= 0x20u;
  }
  else
  {
    if ( a3 != 6 )
      goto LABEL_22;
    v5 = *(_BYTE *)(a2 + 140);
    for ( i = a2; v5 == 12; v5 = *(_BYTE *)(i + 140) )
      i = *(_QWORD *)(i + 160);
    if ( (unsigned __int8)(v5 - 9) <= 2u )
    {
      *(_BYTE *)(*(_QWORD *)(i + 168) + 111LL) |= 1u;
      goto LABEL_9;
    }
    if ( v5 != 2 || (*(_BYTE *)(i + 161) & 8) == 0 )
LABEL_22:
      sub_721090(a1);
    **(_BYTE **)(i + 176) |= 2u;
  }
LABEL_9:
  if ( *(_BYTE *)(a1 + 8) && *(_QWORD *)(a1 + 32) )
  {
    v7 = sub_72A270(a2, a3);
    sub_5CF8B0(12, v7, *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL), *(_QWORD *)(a1 + 32) + 24LL);
  }
  return a2;
}
