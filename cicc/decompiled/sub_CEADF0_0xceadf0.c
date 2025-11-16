// Function: sub_CEADF0
// Address: 0xceadf0
//
__int64 *sub_CEADF0()
{
  if ( byte_4F85230 )
    return &qword_4F85240;
  if ( (unsigned int)sub_2207590(&byte_4F85230) )
  {
    qword_4F85268 = 0x100000000LL;
    qword_4F85248 = 0;
    qword_4F85250 = 0;
    qword_4F85240 = (__int64)&unk_49DD638;
    qword_4F85258 = 0;
    qword_4F85260 = 0;
    sub_CB5980((__int64)&qword_4F85240, 0, 0, 0);
    __cxa_atexit((void (*)(void *))sub_CEAC20, &qword_4F85240, &qword_4A427C0);
    sub_2207640(&byte_4F85230);
  }
  return &qword_4F85240;
}
