// Function: sub_858B20
// Address: 0x858b20
//
unsigned __int64 sub_858B20()
{
  unsigned __int64 result; // rax

  result = dword_4D03C90;
  if ( dword_4D03C90 )
  {
    if ( dword_4D03C88 != dword_4B6EB18 )
    {
      result = dword_4D03C88 - (unsigned __int64)dword_4B6EB18;
      if ( result )
        return result;
      goto LABEL_4;
    }
    result = unk_4D03C8C - (unsigned __int64)unk_4B6EB1C;
    if ( unk_4D03C8C == (unsigned __int64)unk_4B6EB1C )
    {
LABEL_4:
      result = (unsigned __int64)&dword_4D03C84;
      dword_4D03C84 = 1;
    }
  }
  return result;
}
