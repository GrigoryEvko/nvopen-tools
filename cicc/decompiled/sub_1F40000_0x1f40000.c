// Function: sub_1F40000
// Address: 0x1f40000
//
__int64 __fastcall sub_1F40000(char a1, __int64 a2, char a3)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 9:
      result = 234;
      if ( a3 != 5 )
      {
        result = 235;
        if ( a3 != 6 )
        {
          result = 462;
          if ( a3 == 7 )
            return 236;
        }
      }
      break;
    case 10:
      result = 237;
      if ( a3 != 5 )
      {
        result = 238;
        if ( a3 != 6 )
        {
          result = 462;
          if ( a3 == 7 )
            return 239;
        }
      }
      break;
    case 11:
      result = 240;
      if ( a3 != 5 )
      {
        result = 241;
        if ( a3 != 6 )
        {
          result = 462;
          if ( a3 == 7 )
            return 242;
        }
      }
      break;
    case 12:
      result = 243;
      if ( a3 != 5 )
      {
        result = 244;
        if ( a3 != 6 )
        {
          result = 462;
          if ( a3 == 7 )
            return 245;
        }
      }
      break;
    default:
      result = 462;
      if ( a1 == 13 )
      {
        result = 246;
        if ( a3 != 5 )
        {
          result = 247;
          if ( a3 != 6 )
          {
            result = 462;
            if ( a3 == 7 )
              return 248;
          }
        }
      }
      break;
  }
  return result;
}
