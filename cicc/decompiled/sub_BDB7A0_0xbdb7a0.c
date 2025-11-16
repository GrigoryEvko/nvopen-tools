// Function: sub_BDB7A0
// Address: 0xbdb7a0
//
__int64 __fastcall sub_BDB7A0(__int64 a1, __int64 a2)
{
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    return sub_B91C10(a1, a2);
  else
    return 0;
}
