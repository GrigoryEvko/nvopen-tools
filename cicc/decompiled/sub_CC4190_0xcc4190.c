// Function: sub_CC4190
// Address: 0xcc4190
//
__int64 __fastcall sub_CC4190(__int64 a1, __int64 a2)
{
  switch ( a2 )
  {
    case 3LL:
      if ( *(_WORD *)a1 == 28770 && *(_BYTE *)(a1 + 2) == 102 )
        return 8;
      return 0;
    case 6LL:
      if ( *(_DWORD *)a1 != 1600548962 || *(_WORD *)(a1 + 4) != 25954 )
      {
        if ( *(_DWORD *)a1 == 1600548962 && *(_WORD *)(a1 + 4) == 25964 )
          return 8;
        return 0;
      }
      break;
    case 5LL:
      if ( *(_DWORD *)a1 != 1701212258 || *(_BYTE *)(a1 + 4) != 98 )
      {
        if ( *(_DWORD *)a1 == 1701212258 && *(_BYTE *)(a1 + 4) == 108 )
          return 8;
        return 0;
      }
      break;
    default:
      return 0;
  }
  return 9;
}
