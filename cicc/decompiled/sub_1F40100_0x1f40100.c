// Function: sub_1F40100
// Address: 0x1f40100
//
__int64 __fastcall sub_1F40100(char a1, __int64 a2, char a3)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 9:
      result = 249;
      if ( a3 != 5 )
      {
        result = 250;
        if ( a3 != 6 )
        {
          result = 462;
          if ( a3 == 7 )
            return 251;
        }
      }
      break;
    case 10:
      result = 252;
      if ( a3 != 5 )
      {
        result = 253;
        if ( a3 != 6 )
        {
          result = 462;
          if ( a3 == 7 )
            return 254;
        }
      }
      break;
    case 11:
      result = 255;
      if ( a3 != 5 )
      {
        result = 256;
        if ( a3 != 6 )
        {
          result = 462;
          if ( a3 == 7 )
            return 257;
        }
      }
      break;
    case 12:
      result = 258;
      if ( a3 != 5 )
      {
        result = 259;
        if ( a3 != 6 )
        {
          result = 462;
          if ( a3 == 7 )
            return 260;
        }
      }
      break;
    default:
      result = 462;
      if ( a1 == 13 )
      {
        result = 261;
        if ( a3 != 5 )
        {
          result = 262;
          if ( a3 != 6 )
          {
            result = 462;
            if ( a3 == 7 )
              return 263;
          }
        }
      }
      break;
  }
  return result;
}
