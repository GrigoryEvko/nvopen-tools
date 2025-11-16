// Function: sub_C52410
// Address: 0xc52410
//
void *sub_C52410()
{
  if ( byte_4F83CF8 )
    return &unk_4F83D00;
  if ( (unsigned int)sub_2207590(&byte_4F83CF8) )
  {
    dword_4F83D08 = 0;
    qword_4F83D18 = (__int64)&dword_4F83D08;
    qword_4F83D10 = 0;
    qword_4F83D20 = (__int64)&dword_4F83D08;
    qword_4F83D28 = 0;
    __cxa_atexit((void (*)(void *))sub_C4FEF0, &dword_4F83D08 - 2, &qword_4A427C0);
    sub_2207640(&byte_4F83CF8);
  }
  return &unk_4F83D00;
}
