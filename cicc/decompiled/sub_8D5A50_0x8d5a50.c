// Function: sub_8D5A50
// Address: 0x8d5a50
//
__int64 __fastcall sub_8D5A50(__int64 a1)
{
  __int64 v1; // rbx
  char i; // al
  __int64 *j; // r12
  char v5; // al
  __int64 v6; // r12

  v1 = sub_8D4130(a1);
  for ( i = *(_BYTE *)(v1 + 140); i == 12; i = *(_BYTE *)(v1 + 140) )
    v1 = *(_QWORD *)(v1 + 160);
  if ( (unsigned __int8)(i - 9) > 2u )
    return 0;
  if ( !(unsigned int)sub_8D5940(v1, 1, 0) && *(char *)(*(_QWORD *)(*(_QWORD *)v1 + 96LL) + 179LL) < 0 )
  {
    for ( j = **(__int64 ***)(v1 + 168); j; j = (__int64 *)*j )
    {
      v5 = *((_BYTE *)j + 96);
      if ( ((v5 & 2) == 0 || (*(_BYTE *)(v1 + 176) & 0x20) == 0) && (v5 & 1) != 0 && !(unsigned int)sub_8D5A50(j[5]) )
        return 0;
    }
    v6 = *(_QWORD *)(v1 + 160);
    if ( *(_BYTE *)(v1 + 140) == 11 )
    {
      while ( v6 )
      {
        if ( (*(_BYTE *)(v6 + 145) & 0x20) != 0 )
          return 1;
        v6 = *(_QWORD *)(v6 + 112);
      }
      return 0;
    }
    while ( v6 )
    {
      if ( (*(_WORD *)(v6 + 144) & 0x2040) == 0 && !(unsigned int)sub_8D5A50(*(_QWORD *)(v6 + 120)) )
        return 0;
      v6 = *(_QWORD *)(v6 + 112);
    }
  }
  return 1;
}
