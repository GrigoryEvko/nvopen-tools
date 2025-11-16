// Function: sub_6E4B50
// Address: 0x6e4b50
//
__int64 sub_6E4B50()
{
  __int64 result; // rax

  result = word_4D04898;
  if ( word_4D04898 )
  {
    if ( !qword_4F04C50 )
      return (unsigned int)*(char *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 11) >> 31;
    if ( (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 193LL) & 2) == 0 )
      return (unsigned int)*(char *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 11) >> 31;
    result = 1;
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
      return (unsigned int)*(char *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 11) >> 31;
  }
  return result;
}
