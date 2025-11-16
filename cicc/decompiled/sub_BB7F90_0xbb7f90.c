// Function: sub_BB7F90
// Address: 0xbb7f90
//
__int64 *sub_BB7F90()
{
  if ( byte_4F82228 )
    return &qword_4F82240;
  if ( (unsigned int)sub_2207590(&byte_4F82228) )
  {
    qword_4F82248 = 0x7FFFFFFF;
    dword_4F82278 = 0;
    qword_4F82280 = 0;
    qword_4F82240 = (__int64)&unk_49DACB8;
    qword_4F82250 = (__int64)&unk_4F82260;
    qword_4F82258 = 0x400000000LL;
    qword_4F82288 = (__int64)&dword_4F82278;
    qword_4F82290 = (__int64)&dword_4F82278;
    qword_4F82298 = 0;
    __cxa_atexit((void (*)(void *))sub_BB7700, &dword_4F82278 - 14, &qword_4A427C0);
    sub_2207640(&byte_4F82228);
  }
  return &qword_4F82240;
}
