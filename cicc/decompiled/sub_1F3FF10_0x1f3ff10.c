// Function: sub_1F3FF10
// Address: 0x1f3ff10
//
__int64 __fastcall sub_1F3FF10(char a1, __int64 a2, char a3)
{
  __int64 result; // rax

  switch ( a3 )
  {
    case 8:
      result = 221;
      if ( a1 != 9 )
      {
        result = 222;
        if ( a1 != 10 )
        {
          result = 223;
          if ( a1 != 11 )
          {
            result = 224;
            if ( a1 != 12 )
            {
              result = 462;
              if ( a1 == 13 )
                return 225;
            }
          }
        }
      }
      break;
    case 9:
      result = 226;
      if ( a1 != 10 )
      {
        result = 227;
        if ( a1 != 11 )
        {
          result = 228;
          if ( a1 != 12 )
          {
            result = 462;
            if ( a1 == 13 )
              return 229;
          }
        }
      }
      break;
    case 10:
      result = 230;
      if ( a1 != 11 )
      {
        result = 231;
        if ( a1 != 12 )
        {
          result = 462;
          if ( a1 == 13 )
            return 232;
        }
      }
      break;
    default:
      if ( a1 == 12 && a3 == 11 )
        return 233;
      else
        return 462;
  }
  return result;
}
