// Function: sub_BCB090
// Address: 0xbcb090
//
__int64 __fastcall sub_BCB090(__int64 a1)
{
  int i; // eax
  unsigned int v2; // r8d

  for ( i = *(unsigned __int8 *)(a1 + 8); (unsigned int)(i - 17) <= 1; i = *(unsigned __int8 *)(a1 + 8) )
    a1 = *(_QWORD *)(a1 + 24);
  v2 = 11;
  if ( i )
  {
    switch ( i )
    {
      case 1:
        return 8;
      case 2:
        return 24;
      case 3:
        return 53;
      case 4:
        return 64;
      default:
        v2 = -1;
        if ( i == 5 )
          return 113;
        break;
    }
  }
  return v2;
}
