// Function: sub_BCBA50
// Address: 0xbcba50
//
__int64 __fastcall sub_BCBA50(__int64 a1)
{
  unsigned int v1; // eax
  _BOOL4 v2; // edx

  v1 = *(unsigned __int8 *)(a1 + 8);
  v2 = (((_BYTE)v1 - 11) & 0xFD) != 0;
  v1 -= 7;
  LOBYTE(v1) = (unsigned __int8)v1 > 2u;
  return v2 & v1;
}
