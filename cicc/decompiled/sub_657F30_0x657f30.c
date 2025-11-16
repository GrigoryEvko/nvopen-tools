// Function: sub_657F30
// Address: 0x657f30
//
__int64 __fastcall sub_657F30(unsigned int *a1)
{
  __int64 result; // rax
  int v2; // r8d

  result = (unsigned int)qword_4F077B4;
  if ( dword_4F077BC )
  {
    if ( !(_DWORD)qword_4F077B4 )
    {
      if ( qword_4F077A8 <= 0x1116Fu )
        return result;
      goto LABEL_8;
    }
  }
  else if ( !(_DWORD)qword_4F077B4 )
  {
    return result;
  }
  result = 0;
  if ( dword_4F077C4 == 2 && qword_4F077A0 > 0x78B3u )
  {
LABEL_8:
    v2 = sub_729F80(*a1);
    result = 1;
    if ( !v2 )
    {
      sub_684B30(3033, a1);
      dword_4D04820 = 1;
      return 1;
    }
  }
  return result;
}
