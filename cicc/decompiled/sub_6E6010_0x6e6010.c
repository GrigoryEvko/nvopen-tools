// Function: sub_6E6010
// Address: 0x6e6010
//
__int64 sub_6E6010()
{
  unsigned int v0; // r8d

  v0 = 0;
  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
  {
    v0 = 1;
    if ( qword_4D03C50 )
    {
      if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) != 0 )
      {
        v0 = unk_4F07734;
        if ( unk_4F07734 )
          return unk_4F07730 == 0;
      }
    }
  }
  return v0;
}
