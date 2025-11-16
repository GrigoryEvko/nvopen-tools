// Function: sub_1F3FE80
// Address: 0x1f3fe80
//
__int64 __fastcall sub_1F3FE80(char a1, __int64 a2, char a3)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 8:
      result = 462;
      if ( a3 == 9 )
        return 220;
      break;
    case 9:
      result = 219;
      if ( a3 != 10 )
      {
        result = 218;
        if ( a3 != 12 )
        {
          result = 462;
          if ( a3 == 13 )
            return 214;
        }
      }
      break;
    case 10:
      result = 217;
      if ( a3 != 12 )
      {
        result = 462;
        if ( a3 == 13 )
          return 215;
      }
      break;
    default:
      if ( a3 == 12 && a1 == 11 )
        return 216;
      else
        return 462;
  }
  return result;
}
