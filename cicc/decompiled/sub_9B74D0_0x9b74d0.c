// Function: sub_9B74D0
// Address: 0x9b74d0
//
__int64 __fastcall sub_9B74D0(unsigned int a1, __int64 a2)
{
  unsigned int v2; // r13d

  v2 = sub_9B7470(a1);
  if ( (_BYTE)v2 )
    return v2;
  if ( a2 && (unsigned __int8)sub_B60C40(a1) )
    return sub_DFAA10(a2, a1);
  if ( a1 > 0x174 )
    return v2;
  if ( a1 > 0x14C )
  {
    switch ( a1 )
    {
      case 0x14Du:
      case 0x153u:
      case 0x168u:
      case 0x171u:
      case 0x174u:
        return 1;
      default:
        return v2;
    }
  }
  if ( a1 != 179 )
  {
    LOBYTE(v2) = a1 == 312;
    return v2;
  }
  return 1;
}
