// Function: sub_6E94D0
// Address: 0x6e94d0
//
__int64 sub_6E94D0()
{
  __int64 result; // rax

  result = 32;
  if ( !unk_4D04000 )
  {
    result = 848;
    if ( dword_4F077C4 == 2 )
    {
      result = 2138;
      if ( unk_4F07778 <= 201102 )
        return dword_4F07774 == 0 ? 848 : 2138;
    }
  }
  return result;
}
