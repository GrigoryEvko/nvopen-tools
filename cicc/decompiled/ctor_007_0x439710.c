// Function: ctor_007
// Address: 0x439710
//
int ctor_007()
{
  sub_2208040(&unk_4F6D2DD);
  __cxa_atexit(sub_2208810, &unk_4F6D2DD, &qword_4A427C0);
  dword_4F6D2A8 = 0;
  qword_4F6D2B0 = 0;
  qword_4F6D2B8 = (__int64)&dword_4F6D2A8;
  qword_4F6D2C0 = (__int64)&dword_4F6D2A8;
  qword_4F6D2C8 = 0;
  return __cxa_atexit(sub_8FCA20, &dword_4F6D2A8 - 2, &qword_4A427C0);
}
