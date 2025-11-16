// Function: sub_71B520
// Address: 0x71b520
//
unsigned __int64 __fastcall sub_71B520(__int64 a1)
{
  unsigned __int64 result; // rax

  result = (unsigned __int64)&dword_4D0488C;
  if ( !dword_4D0488C )
  {
    result = (unsigned __int64)&word_4D04898;
    if ( word_4D04898 )
    {
      result = (unsigned int)qword_4F077B4;
      if ( (_DWORD)qword_4F077B4 )
      {
        result = (unsigned __int64)&qword_4F077A0;
        if ( qword_4F077A0 > 0x765Bu )
          result = sub_729F80(dword_4F063F8);
      }
    }
  }
  *(_BYTE *)(a1 + 29) |= 8u;
  return result;
}
