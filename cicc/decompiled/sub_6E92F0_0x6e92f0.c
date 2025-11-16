// Function: sub_6E92F0
// Address: 0x6e92f0
//
__int64 sub_6E92F0()
{
  __int64 result; // rax

  result = 31;
  if ( !unk_4D04000 )
  {
    result = 847;
    if ( dword_4F077C4 == 2 )
    {
      result = 2140;
      if ( unk_4F07778 <= 201102 )
        return dword_4F07774 == 0 ? 847 : 2140;
    }
  }
  return result;
}
