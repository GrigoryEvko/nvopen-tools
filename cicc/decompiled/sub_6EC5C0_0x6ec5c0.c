// Function: sub_6EC5C0
// Address: 0x6ec5c0
//
__int64 __fastcall sub_6EC5C0(__int64 a1, _QWORD *a2)
{
  char i; // al
  unsigned int v3; // r9d
  __int64 *v4; // rax
  __int64 *v5; // r12

  for ( i = *(_BYTE *)(a1 + 140); i == 12; i = *(_BYTE *)(a1 + 140) )
    a1 = *(_QWORD *)(a1 + 160);
  if ( (unsigned __int8)(i - 9) > 2u )
  {
    v3 = 0;
    if ( i != 2 || (*(_BYTE *)(a1 + 161) & 8) == 0 || (**(_BYTE **)(a1 + 176) & 2) == 0 )
      return v3;
  }
  else
  {
    v3 = 0;
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 168) + 111LL) & 1) == 0 )
      return v3;
  }
  if ( a2 )
  {
    v4 = sub_5CF860(12, a1);
    v5 = v4;
    if ( v4 )
    {
      if ( (unsigned int)sub_72AE80(*(_QWORD *)(v4[4] + 40)) )
        *a2 = *(_QWORD *)(*(_QWORD *)(v5[4] + 40) + 184LL);
    }
  }
  return 1;
}
