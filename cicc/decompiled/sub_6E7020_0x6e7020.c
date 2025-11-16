// Function: sub_6E7020
// Address: 0x6e7020
//
__int64 __fastcall sub_6E7020(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax
  char i; // dl

  result = sub_6E6A50(a1, (__int64)a2);
  if ( a2[16] )
  {
    result = *(_QWORD *)a2;
    for ( i = *(_BYTE *)(*(_QWORD *)a2 + 140LL); i == 12; i = *(_BYTE *)(result + 140) )
      result = *(_QWORD *)(result + 160);
    if ( i )
      a2[19] |= 0x10u;
  }
  return result;
}
