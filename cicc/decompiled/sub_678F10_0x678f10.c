// Function: sub_678F10
// Address: 0x678f10
//
__int64 sub_678F10()
{
  __int64 result; // rax

  if ( !unk_4D042DC )
  {
    byte_4CFDE80 = 5;
    goto LABEL_3;
  }
  if ( dword_4D04964 && !unk_4D04298 )
  {
    byte_4CFDE80 = 8;
LABEL_3:
    result = unk_4D04508;
    if ( !unk_4D04508 )
      return result;
    return sub_8539C0(&off_4A441E0);
  }
  byte_4CFDE80 = 10;
  result = unk_4D04508;
  if ( unk_4D04508 )
    return sub_8539C0(&off_4A441E0);
  return result;
}
