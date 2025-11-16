// Function: sub_2FE48D0
// Address: 0x2fe48d0
//
__int64 __fastcall sub_2FE48D0(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 result; // rax

  v2 = sub_AE2980(a2, 0)[1];
  result = 2;
  if ( v2 != 1 )
  {
    result = 3;
    if ( v2 != 2 )
    {
      result = 4;
      if ( v2 != 4 )
      {
        result = 5;
        if ( v2 != 8 )
        {
          result = 6;
          if ( v2 != 16 )
          {
            result = 7;
            if ( v2 != 32 )
            {
              result = 8;
              if ( v2 != 64 )
                return 9 * (unsigned int)(v2 == 128);
            }
          }
        }
      }
    }
  }
  return result;
}
