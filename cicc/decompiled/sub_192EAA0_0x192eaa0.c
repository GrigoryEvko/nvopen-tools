// Function: sub_192EAA0
// Address: 0x192eaa0
//
__int64 *sub_192EAA0()
{
  if ( byte_4FAF328 )
    return &qword_4FAF340;
  if ( (unsigned int)sub_2207590(&byte_4FAF328) )
  {
    qword_4FAF350 = 1;
    qword_4FAF340 = (__int64)&qword_4FAF350;
    qword_4FAF370 = (__int64)&unk_4FAF380;
    qword_4FAF378 = 0x400000000LL;
    qword_4FAF348 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_192D0E0, &qword_4FAF340, &qword_4A427C0);
    sub_2207640(&byte_4FAF328);
  }
  return &qword_4FAF340;
}
