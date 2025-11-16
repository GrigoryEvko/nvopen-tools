// Function: sub_688C20
// Address: 0x688c20
//
__int64 __fastcall sub_688C20(__int64 a1, FILE *a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 v5; // r12
  __int64 i; // r13

  if ( (*(_BYTE *)(a1 + 17) & 0x20) != 0 )
    return 0;
  v5 = a3;
  if ( !(unsigned int)sub_8D3A70(a3) )
    return 0;
  for ( i = *(_QWORD *)(a1 + 24); *(_BYTE *)(v5 + 140) == 12; v5 = *(_QWORD *)(v5 + 160) )
    ;
  if ( (*(_BYTE *)(i + 81) & 0x10) == 0 )
  {
    if ( (unsigned int)sub_6E5430() )
      sub_6854C0(0x3FEu, a2, i);
    return 0;
  }
  if ( (*(_BYTE *)(v5 + 177) & 0x20) == 0 && (*(_BYTE *)(*(_QWORD *)(i + 64) + 177LL) & 0x20) == 0 )
  {
    v3 = sub_8D5DF0(v5);
    if ( !v3 )
    {
      if ( (unsigned int)sub_6E5430() )
      {
        sub_685360(0xF4u, a2, v5);
        return v3;
      }
      return 0;
    }
  }
  return 1;
}
