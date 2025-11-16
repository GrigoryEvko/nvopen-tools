// Function: sub_7328C0
// Address: 0x7328c0
//
__int64 __fastcall sub_7328C0(__int64 a1)
{
  __int64 **v2; // rax

  if ( *(_BYTE *)(a1 + 140) != 7 )
    return 0;
  v2 = **(__int64 ****)(a1 + 168);
  if ( !v2 )
    return 0;
  while ( ((_BYTE)v2[4] & 4) == 0 || v2[5] )
  {
    v2 = (__int64 **)*v2;
    if ( !v2 )
      return 0;
  }
  return 1;
}
