// Function: sub_13D1D50
// Address: 0x13d1d50
//
__int64 __fastcall sub_13D1D50(unsigned __int8 *a1, __int64 a2, char a3, _QWORD *a4)
{
  unsigned __int8 v5; // al
  __int64 v6; // r13

  v5 = a1[16];
  if ( v5 <= 0x10u )
  {
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
      v6 = sub_14D6F90(22, a1, a2, *a4);
      if ( v6 )
        return v6;
      v5 = a1[16];
    }
    if ( v5 == 9 )
      goto LABEL_8;
  }
  if ( *(_BYTE *)(a2 + 16) == 9 )
LABEL_8:
    v6 = sub_15A11D0(*(_QWORD *)a1, 0, 0);
  else
    v6 = sub_13CDA40(a1, (_QWORD *)a2);
  if ( v6 || (a3 & 2) == 0 )
    return v6;
  if ( (unsigned __int8)sub_13CBF20((__int64)a1) )
    return sub_15A06D0(*(_QWORD *)a1);
  if ( !(unsigned __int8)sub_13CC390((__int64)a1) )
    return v6;
  return sub_15A1390(*(_QWORD *)a1);
}
