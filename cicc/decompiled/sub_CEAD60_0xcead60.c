// Function: sub_CEAD60
// Address: 0xcead60
//
__int64 *sub_CEAD60()
{
  if ( byte_4F85288 )
    return &qword_4F85290;
  if ( (unsigned int)sub_2207590(&byte_4F85288) )
  {
    sub_C94DE0((__int64)&qword_4F85290);
    byte_4F852A0 = 0;
    qword_4F85290 = (__int64)&unk_49DD618;
    __cxa_atexit((void (*)(void *))sub_CEAAA0, &qword_4F85290, &qword_4A427C0);
    sub_2207640(&byte_4F85288);
  }
  return &qword_4F85290;
}
