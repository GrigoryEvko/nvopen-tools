// Function: sub_2D42F30
// Address: 0x2d42f30
//
__int64 __fastcall sub_2D42F30(__int64 a1, __int64 a2, unsigned int a3)
{
  int v3; // edx
  __int64 result; // rax

  v3 = sub_AE2980(a2, a3)[1];
  result = 2;
  if ( v3 != 1 )
  {
    result = 3;
    if ( v3 != 2 )
    {
      result = 4;
      if ( v3 != 4 )
      {
        result = 5;
        if ( v3 != 8 )
        {
          result = 6;
          if ( v3 != 16 )
          {
            result = 7;
            if ( v3 != 32 )
            {
              result = 8;
              if ( v3 != 64 )
                return 9 * (unsigned int)(v3 == 128);
            }
          }
        }
      }
    }
  }
  return result;
}
