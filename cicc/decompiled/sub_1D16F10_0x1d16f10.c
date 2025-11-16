// Function: sub_1D16F10
// Address: 0x1d16f10
//
__int64 __fastcall sub_1D16F10(unsigned int a1, unsigned int a2, char a3)
{
  __int64 result; // rax

  if ( !a3 )
  {
    result = a2 | a1;
    if ( (unsigned int)result <= 0x17 )
      return result;
    goto LABEL_11;
  }
  if ( a1 == 17 )
  {
LABEL_17:
    result = a2 | a1;
    if ( (unsigned int)result <= 0x17 )
    {
LABEL_12:
      if ( (_DWORD)result != 14 )
        return result;
      goto LABEL_13;
    }
LABEL_11:
    result = (unsigned int)result & 0xFFFFFFEF;
    goto LABEL_12;
  }
  if ( a1 <= 0x11 )
  {
    if ( a2 != 17 )
    {
      if ( a2 <= 0x11 )
        goto LABEL_8;
      if ( a2 <= 0x15 )
        return 24;
    }
    goto LABEL_17;
  }
  if ( a1 > 0x15 || a2 == 17 )
    goto LABEL_17;
  result = 24;
  if ( a2 <= 0x11 )
    return result;
  if ( a2 > 0x15 )
    goto LABEL_17;
LABEL_8:
  result = a1 | a2;
  if ( (_DWORD)result != 14 )
    return result;
LABEL_13:
  if ( a3 )
    return 22;
  return result;
}
