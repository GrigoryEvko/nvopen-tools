// Function: sub_3366D50
// Address: 0x3366d50
//
__int64 __fastcall sub_3366D50(__int64 a1, unsigned int a2)
{
  int v2; // edx
  __int64 result; // rax

  v2 = sub_AE2980(a1, a2)[1];
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
