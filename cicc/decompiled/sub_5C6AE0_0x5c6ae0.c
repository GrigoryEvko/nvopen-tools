// Function: sub_5C6AE0
// Address: 0x5c6ae0
//
__int64 __fastcall sub_5C6AE0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  sub_6851C0(3560, a1 + 56);
  if ( a3 == 11 )
    *(_BYTE *)(a2 + 196) |= 8u;
  else
    *(_BYTE *)(a2 + 156) |= 0x80u;
  sub_7604D0(a2, a3);
  return a2;
}
