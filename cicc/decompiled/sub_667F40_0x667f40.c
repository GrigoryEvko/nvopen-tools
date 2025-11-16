// Function: sub_667F40
// Address: 0x667f40
//
__int64 __fastcall sub_667F40(char a1)
{
  __int64 result; // rax

  result = 9;
  if ( a1 != 9 )
  {
    if ( !a1 || a1 == 10 )
    {
      return 10;
    }
    else
    {
      result = 11;
      if ( a1 != 1 )
      {
        if ( a1 == 2 || a1 == 11 )
        {
          return 12;
        }
        else
        {
          result = 14;
          if ( a1 != 3 )
          {
            result = 15;
            if ( a1 != 4 )
            {
              result = 17;
              if ( a1 != 5 )
              {
                result = 18;
                if ( a1 != 7 )
                {
                  if ( a1 == 8 )
                    return 19;
                  result = 15;
                  if ( a1 == 13 )
                    return 19;
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
