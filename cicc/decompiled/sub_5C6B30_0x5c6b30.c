// Function: sub_5C6B30
// Address: 0x5c6b30
//
__int64 __fastcall sub_5C6B30(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  if ( a3 == 11 )
  {
    *(_BYTE *)(a2 + 201) |= 2u;
  }
  else
  {
    if ( a3 != 7 )
      sub_721090(a1);
    *(_BYTE *)(a2 + 168) |= 0x40u;
  }
  sub_7604D0(a2, a3);
  return a2;
}
