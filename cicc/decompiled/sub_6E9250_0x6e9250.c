// Function: sub_6E9250
// Address: 0x6e9250
//
_BOOL8 __fastcall sub_6E9250(_DWORD *a1)
{
  _DWORD *v2; // rsi

  if ( dword_4D0488C )
    return 0;
  v2 = a1;
  if ( word_4D04898
    && (_DWORD)qword_4F077B4
    && qword_4F077A0 > 0x765Bu
    && (v2 = a1, (unsigned int)sub_729F80(dword_4F063F8)) )
  {
    return 0;
  }
  else
  {
    return (unsigned int)sub_6E91E0(0x39u, v2) != 0;
  }
}
