// Function: sub_165C1A0
// Address: 0x165c1a0
//
unsigned __int64 __fastcall sub_165C1A0(__int64 a1)
{
  unsigned __int8 v1; // al

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 <= 0x17u )
    return 0;
  if ( v1 == 78 )
    return a1 | 4;
  if ( v1 != 29 )
    return 0;
  return a1 & 0xFFFFFFFFFFFFFFFBLL;
}
