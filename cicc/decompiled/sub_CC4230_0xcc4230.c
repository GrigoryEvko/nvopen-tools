// Function: sub_CC4230
// Address: 0xcc4230
//
__int64 __fastcall sub_CC4230(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  switch ( a2 )
  {
    case 5LL:
      if ( *(_DWORD *)a1 == 1819308129 && *(_BYTE *)(a1 + 4) == 101 )
        return 1;
      if ( *(_DWORD *)a1 == 1702129257 && *(_BYTE *)(a1 + 4) == 108 )
        return 14;
      return 0;
    case 2LL:
      result = 2;
      if ( *(_WORD *)a1 != 25456 )
      {
        result = 13;
        if ( *(_WORD *)a1 != 25967 )
          return 0;
      }
      break;
    case 4LL:
      result = 3;
      if ( *(_DWORD *)a1 != 1768252275 )
      {
        result = 11;
        if ( *(_DWORD *)a1 != 1634952557 )
        {
          result = 12;
          if ( *(_DWORD *)a1 != 1702065523 )
            return 0;
        }
      }
      break;
    case 3LL:
      if ( *(_WORD *)a1 == 26995 && *(_BYTE *)(a1 + 2) == 101 )
      {
        return 3;
      }
      else if ( *(_WORD *)a1 == 29542 && *(_BYTE *)(a1 + 2) == 108 )
      {
        return 4;
      }
      else if ( *(_WORD *)a1 == 25193 && *(_BYTE *)(a1 + 2) == 109 )
      {
        return 5;
      }
      else if ( *(_WORD *)a1 == 28009 && *(_BYTE *)(a1 + 2) == 103 )
      {
        return 6;
      }
      else if ( *(_WORD *)a1 == 29805 && *(_BYTE *)(a1 + 2) == 105 )
      {
        return 7;
      }
      else if ( *(_WORD *)a1 == 29539 && *(_BYTE *)(a1 + 2) == 114 )
      {
        return 9;
      }
      else
      {
        if ( *(_WORD *)a1 != 28001 || *(_BYTE *)(a1 + 2) != 100 )
          return 0;
        return 10;
      }
    default:
      if ( a2 == 6 && *(_DWORD *)a1 == 1684633198 && *(_WORD *)(a1 + 4) == 24937 )
        return 8;
      return 0;
  }
  return result;
}
