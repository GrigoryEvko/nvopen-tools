// Function: sub_1471280
// Address: 0x1471280
//
__int64 __fastcall sub_1471280(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d

  if ( *(_BYTE *)(a2 + 16) == 5 )
    return 0;
  if ( (unsigned __int8)sub_15F2370(a2) )
  {
    v2 = (unsigned __int8)sub_15F2380(a2) == 0 ? 2 : 6;
  }
  else
  {
    if ( !(unsigned __int8)sub_15F2380(a2) )
      return 0;
    v2 = 4;
  }
  if ( !(unsigned __int8)sub_1471070(a1, a2) )
    return 0;
  return v2;
}
