// Function: sub_BB77E0
// Address: 0xbb77e0
//
__int64 __fastcall sub_BB77E0(__int64 a1, unsigned int *a2)
{
  unsigned int v2; // ebx
  __int64 result; // rax

  v2 = *a2;
  result = (unsigned __int8)byte_4F82228;
  if ( !byte_4F82228 )
  {
    result = sub_2207590(&byte_4F82228);
    if ( (_DWORD)result )
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
      result = sub_2207640(&byte_4F82228);
    }
  }
  qword_4F82248 = v2;
  return result;
}
