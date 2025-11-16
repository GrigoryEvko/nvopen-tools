// Function: sub_2217070
// Address: 0x2217070
//
__int64 __fastcall sub_2217070(__int64 a1, unsigned __int16 a2)
{
  if ( a2 == 2048 )
    return __wctype_l();
  if ( a2 > 0x800u )
  {
    if ( a2 != 4096 )
    {
      if ( a2 <= 0x1000u )
      {
        if ( a2 == 3072 || a2 == 3076 )
          return __wctype_l();
      }
      else if ( a2 == 0x2000 || a2 == 0x4000 )
      {
        return __wctype_l();
      }
      return 0;
    }
    return __wctype_l();
  }
  if ( a2 == 256 )
    return __wctype_l();
  if ( a2 > 0x100u )
  {
    if ( a2 == 512 || a2 == 1024 )
      return __wctype_l();
  }
  else if ( a2 == 2 || a2 == 4 || a2 == 1 )
  {
    return __wctype_l();
  }
  return 0;
}
