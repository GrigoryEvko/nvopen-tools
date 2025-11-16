// Function: sub_1F7DE30
// Address: 0x1f7de30
//
__int64 __fastcall sub_1F7DE30(_QWORD *a1, unsigned int a2)
{
  unsigned __int8 v2; // dl

  if ( a2 == 32 )
    return 5;
  if ( a2 <= 0x20 )
  {
    if ( a2 == 8 )
    {
      return 3;
    }
    else
    {
      v2 = 4;
      if ( a2 != 16 )
      {
        v2 = 2;
        if ( a2 != 1 )
          return sub_1F58CC0(a1, a2);
      }
    }
    return v2;
  }
  if ( a2 == 64 )
    return 6;
  if ( a2 == 128 )
    return 7;
  return sub_1F58CC0(a1, a2);
}
