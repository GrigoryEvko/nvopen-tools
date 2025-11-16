// Function: sub_CB7330
// Address: 0xcb7330
//
__int64 *sub_CB7330()
{
  if ( byte_4F85000 )
    return &qword_4F85020;
  if ( (unsigned int)sub_2207590(&byte_4F85000) )
  {
    qword_4F85028 = 0;
    qword_4F85030 = 0;
    qword_4F85038 = 0;
    qword_4F85040 = 0;
    qword_4F85048 = 0;
    qword_4F85020 = (__int64)&unk_49DD308;
    __cxa_atexit((void (*)(void *))sub_CB58D0, &qword_4F85020, &qword_4A427C0);
    sub_2207640(&byte_4F85000);
  }
  return &qword_4F85020;
}
