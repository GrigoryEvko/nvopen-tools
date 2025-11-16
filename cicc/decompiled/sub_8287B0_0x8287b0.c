// Function: sub_8287B0
// Address: 0x8287b0
//
__int64 __fastcall sub_8287B0(_BYTE *a1)
{
  __int64 result; // rax

  if ( (a1[81] & 0x10) != 0 )
    return 1;
  if ( (unsigned int)sub_880A60() )
  {
    if ( !dword_4F077BC )
      return 1;
    result = 0;
    if ( qword_4F077A8 > 0x76BFu && !*(_BYTE *)(*(_QWORD *)a1 + 72LL) )
    {
      if ( qword_4F077A8 > 0x9E33u
        || unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
      {
        return 1;
      }
      result = 0;
      if ( dword_4F04C44 != -1 )
        return 1;
    }
  }
  else
  {
    result = dword_4F077BC;
    if ( dword_4F077BC )
    {
      result = 0;
      if ( qword_4F077A8 <= 0x76BFu )
        return (unsigned int)sub_8809D0(a1) != 0;
    }
  }
  return result;
}
