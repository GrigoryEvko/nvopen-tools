// Function: sub_10E0640
// Address: 0x10e0640
//
__int64 __fastcall sub_10E0640(unsigned int a1)
{
  if ( a1 != 365 )
  {
    if ( a1 > 0x16D )
    {
      if ( a1 == 366 )
        return 36;
    }
    else
    {
      if ( a1 == 329 )
        return 38;
      if ( a1 == 330 )
        return 40;
    }
    BUG();
  }
  return 34;
}
