// Function: sub_1A82450
// Address: 0x1a82450
//
bool __fastcall sub_1A82450(__int64 a1)
{
  int v1; // ecx
  unsigned int v3; // ecx

  v1 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)v1 > 0x17u )
  {
    v3 = v1 - 24;
  }
  else
  {
    if ( (_BYTE)v1 != 5 )
      return 0;
    v3 = *(unsigned __int16 *)(a1 + 18);
  }
  if ( v3 > 0x37 )
    return 0;
  return ((1LL << v3) & 0xA1800100000000LL) != 0;
}
