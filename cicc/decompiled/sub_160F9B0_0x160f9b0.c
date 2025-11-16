// Function: sub_160F9B0
// Address: 0x160f9b0
//
__int64 sub_160F9B0()
{
  _DWORD *v0; // rax
  unsigned int v1; // r8d

  v0 = (_DWORD *)qword_4F9E980;
  if ( qword_4F9E980 != qword_4F9E988 )
  {
    while ( dword_4F9E8C8 != *v0 )
    {
      if ( (_DWORD *)qword_4F9E988 == ++v0 )
        goto LABEL_7;
    }
    v1 = 0;
    goto LABEL_6;
  }
LABEL_7:
  if ( dword_4F9E8CC < dword_4F9EA60 || dword_4F9EA60 == -1 )
  {
    v1 = 1;
LABEL_6:
    ++dword_4F9E8CC;
    return v1;
  }
  return 0;
}
