// Function: sub_F8F4A0
// Address: 0xf8f4a0
//
bool __fastcall sub_F8F4A0(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int64 v3; // rcx

  if ( *a1 == 61 )
  {
    if ( sub_B46500(a1) || (a1[2] & 1) != 0 )
      return 0;
  }
  else if ( *a1 != 62 || sub_B46500(a1) || (a1[2] & 1) != 0 )
  {
    return 0;
  }
  if ( !(unsigned __int8)sub_DFB150(a2) )
    return 0;
  _BitScanReverse64(&v3, 1LL << (*((_WORD *)a1 + 1) >> 1));
  return 0x8000000000000000LL >> ((unsigned __int8)v3 ^ 0x3Fu) <= 0xFFFFFFFF;
}
