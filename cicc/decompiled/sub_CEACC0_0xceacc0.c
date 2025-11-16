// Function: sub_CEACC0
// Address: 0xceacc0
//
__int64 *sub_CEACC0()
{
  if ( byte_4F852A8 )
    return &qword_4F852C0;
  if ( (unsigned int)sub_2207590(&byte_4F852A8) )
  {
    sub_C94DE0((__int64)&qword_4F852C0);
    qword_4F852C0 = (__int64)&unk_49DD5D8;
    memset(&unk_4F852D0, 0, 0xC8u);
    __cxa_atexit((void (*)(void *))sub_CEAA60, &qword_4F852C0, &qword_4A427C0);
    sub_2207640(&byte_4F852A8);
  }
  return &qword_4F852C0;
}
