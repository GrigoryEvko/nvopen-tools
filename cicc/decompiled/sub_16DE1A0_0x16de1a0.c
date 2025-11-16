// Function: sub_16DE1A0
// Address: 0x16de1a0
//
__int64 __fastcall sub_16DE1A0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  switch ( a2 )
  {
    case 5LL:
      return *(_DWORD *)a1 == 1819308129 && *(_BYTE *)(a1 + 4) == 101;
    case 2LL:
      result = 2;
      if ( *(_WORD *)a1 != 25456 )
        return 16 * (unsigned int)(*(_WORD *)a1 == 25967);
      break;
    case 4LL:
      result = 3;
      if ( *(_DWORD *)a1 != 1768252275 )
      {
        result = 14;
        if ( *(_DWORD *)a1 != 1634952557 )
        {
          result = 15;
          if ( *(_DWORD *)a1 != 1702065523 )
            return 0;
        }
      }
      break;
    case 3LL:
      if ( *(_WORD *)a1 == 26466 && *(_BYTE *)(a1 + 2) == 112 )
        return 4;
      if ( *(_WORD *)a1 == 26466 && *(_BYTE *)(a1 + 2) == 113 )
        return 5;
      if ( *(_WORD *)a1 == 29542 && *(_BYTE *)(a1 + 2) == 108 )
        return 6;
      if ( *(_WORD *)a1 == 25193 && *(_BYTE *)(a1 + 2) == 109 )
        return 7;
      if ( *(_WORD *)a1 == 28009 && *(_BYTE *)(a1 + 2) == 103 )
        return 8;
      if ( *(_WORD *)a1 == 29805 && *(_BYTE *)(a1 + 2) == 105 )
        return 9;
      if ( *(_WORD *)a1 == 29539 && *(_BYTE *)(a1 + 2) == 114 )
        return 11;
      if ( *(_WORD *)a1 == 28001 && *(_BYTE *)(a1 + 2) == 100 )
        return 13;
      return 0;
    case 6LL:
      if ( *(_DWORD *)a1 == 1684633198 && *(_WORD *)(a1 + 4) == 24937 )
      {
        return 10;
      }
      else
      {
        if ( *(_DWORD *)a1 != 1769109869 || *(_WORD *)(a1 + 4) != 25697 )
          return 0;
        return 12;
      }
    default:
      return 0;
  }
  return result;
}
