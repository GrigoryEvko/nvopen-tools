// Function: sub_693A90
// Address: 0x693a90
//
__int64 sub_693A90()
{
  unsigned int v0; // r8d

  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
  {
    if ( qword_4D0495C )
      return (unsigned __int8)(*(_BYTE *)(qword_4D03C50 + 16LL) - 1) <= 2u;
    v0 = dword_4F077BC;
    if ( dword_4F077BC )
    {
      v0 = 0;
      if ( qword_4F077A8 <= 0x76BFu )
        return (unsigned __int8)(*(_BYTE *)(qword_4D03C50 + 16LL) - 1) <= 2u;
    }
  }
  else
  {
    return *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u;
  }
  return v0;
}
