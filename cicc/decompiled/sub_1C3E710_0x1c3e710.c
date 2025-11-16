// Function: sub_1C3E710
// Address: 0x1c3e710
//
__int64 *sub_1C3E710()
{
  if ( byte_4FBA5E8 )
    return &qword_4FBA600;
  if ( (unsigned int)sub_2207590(&byte_4FBA5E8) )
  {
    sub_16D40B0((__int64)&qword_4FBA600);
    qword_4FBA600 = (__int64)&unk_49F7940;
    memset(&unk_4FBA610, 0, 0xC8u);
    __cxa_atexit((void (*)(void *))sub_1C3E4B0, &qword_4FBA600, &qword_4A427C0);
    sub_2207640(&byte_4FBA5E8);
  }
  return &qword_4FBA600;
}
