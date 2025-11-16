// Function: sub_1C3E7B0
// Address: 0x1c3e7b0
//
__int64 *sub_1C3E7B0()
{
  if ( byte_4FBA5C8 )
    return &qword_4FBA5D0;
  if ( (unsigned int)sub_2207590(&byte_4FBA5C8) )
  {
    sub_16D40B0((__int64)&qword_4FBA5D0);
    byte_4FBA5E0 = 0;
    qword_4FBA5D0 = (__int64)&unk_49F7980;
    __cxa_atexit((void (*)(void *))sub_1C3E4F0, &qword_4FBA5D0, &qword_4A427C0);
    sub_2207640(&byte_4FBA5C8);
  }
  return &qword_4FBA5D0;
}
