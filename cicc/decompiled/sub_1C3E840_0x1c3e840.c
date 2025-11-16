// Function: sub_1C3E840
// Address: 0x1c3e840
//
__int64 *sub_1C3E840()
{
  if ( byte_4FBA570 )
    return &qword_4FBA580;
  if ( (unsigned int)sub_2207590(&byte_4FBA570) )
  {
    dword_4FBA5A0 = 1;
    qword_4FBA598 = 0;
    qword_4FBA580 = (__int64)&unk_49F79A0;
    qword_4FBA590 = 0;
    qword_4FBA588 = 0;
    sub_16E7A40((__int64)&qword_4FBA580, 0, 0, 0);
    __cxa_atexit((void (*)(void *))sub_1C3E670, &qword_4FBA580, &qword_4A427C0);
    sub_2207640(&byte_4FBA570);
  }
  return &qword_4FBA580;
}
