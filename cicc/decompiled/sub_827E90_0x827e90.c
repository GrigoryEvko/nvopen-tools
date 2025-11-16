// Function: sub_827E90
// Address: 0x827e90
//
_BOOL8 __fastcall sub_827E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax

  v4 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( dword_4F04C44 == -1 )
  {
    if ( (*(_BYTE *)(v4 + 6) & 2) == 0 || (*(_BYTE *)(v4 + 12) & 0x20) != 0 )
      return 0;
  }
  else if ( (*(_BYTE *)(v4 + 12) & 0x20) != 0 )
  {
    return 0;
  }
  return (unsigned int)sub_866B30(a1, a2, &dword_4F04C44, a4, 0) == 0;
}
