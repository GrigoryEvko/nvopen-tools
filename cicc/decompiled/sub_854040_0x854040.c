// Function: sub_854040
// Address: 0x854040
//
_DWORD *__fastcall sub_854040(__int64 a1)
{
  _QWORD *i; // rax

  for ( i = (_QWORD *)qword_4D03E88; i; i = (_QWORD *)*i )
  {
    if ( !*i )
    {
      *i = a1;
      if ( !qword_4D03E88 )
        goto LABEL_8;
LABEL_6:
      dword_4F061FC = 1;
      return &dword_4F061FC;
    }
  }
  if ( qword_4D03E88 )
    goto LABEL_6;
LABEL_8:
  qword_4D03E88 = a1;
  dword_4F061FC = 1;
  return &dword_4F061FC;
}
