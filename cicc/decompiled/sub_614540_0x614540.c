// Function: sub_614540
// Address: 0x614540
//
unsigned __int64 sub_614540()
{
  unsigned __int64 result; // rax

  result = (unsigned __int64)&dword_4F077C4;
  if ( dword_4F077C4 != 2 )
  {
    result = (unsigned __int64)&unk_4F07778;
    if ( unk_4F07778 > 199900 )
    {
      result = (unsigned __int8)byte_4CF8126;
      LOBYTE(result) = byte_4CF8166 | byte_4CF8165 | byte_4CF8164 | byte_4CF8126;
      if ( (_BYTE)result )
        sub_6849E0(1027);
    }
  }
  return result;
}
