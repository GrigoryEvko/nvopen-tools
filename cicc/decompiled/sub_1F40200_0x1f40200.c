// Function: sub_1F40200
// Address: 0x1f40200
//
__int64 __fastcall sub_1F40200(char a1, __int64 a2, char a3)
{
  __int64 result; // rax

  if ( a1 == 5 )
  {
    result = 264;
    if ( a3 != 9 )
    {
      result = 265;
      if ( a3 != 10 )
      {
        result = 266;
        if ( a3 != 11 )
        {
          result = 267;
          if ( a3 != 12 )
          {
            result = 462;
            if ( a3 == 13 )
              return 268;
          }
        }
      }
    }
  }
  else if ( a1 == 6 )
  {
    result = 269;
    if ( a3 != 9 )
    {
      result = 270;
      if ( a3 != 10 )
      {
        result = 271;
        if ( a3 != 11 )
        {
          result = 272;
          if ( a3 != 12 )
          {
            result = 462;
            if ( a3 == 13 )
              return 273;
          }
        }
      }
    }
  }
  else
  {
    result = 462;
    if ( a1 == 7 )
    {
      result = 274;
      if ( a3 != 9 )
      {
        result = 275;
        if ( a3 != 10 )
        {
          result = 276;
          if ( a3 != 11 )
          {
            result = 277;
            if ( a3 != 12 )
            {
              result = 462;
              if ( a3 == 13 )
                return 278;
            }
          }
        }
      }
    }
  }
  return result;
}
