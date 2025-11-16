// Function: sub_67C430
// Address: 0x67c430
//
_BOOL8 __fastcall sub_67C430(char a1, char a2, _BYTE *a3)
{
  _BOOL8 result; // rax

  result = a2 & 1;
  if ( (a2 & 1) != 0 && (a1 & 1) == 0 )
    goto LABEL_3;
  if ( (a2 & 2) != 0 )
  {
    result = 1;
    if ( (a1 & 2) == 0 )
      goto LABEL_3;
  }
  if ( (a2 & 4) == 0 || (result = 1, (a1 & 4) != 0) )
  {
    if ( (a2 & 8) == 0 || (result = 1, (a1 & 8) != 0) )
    {
      *a3 = ((a1 & 0x10) != 0) + 9;
      return (a2 & 0x10) != 0;
    }
LABEL_3:
    *a3 = ((a1 & 0x10) != 0) + 9;
    return result;
  }
  *a3 = ((a1 & 0x10) != 0) + 9;
  return result;
}
