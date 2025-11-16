// Function: sub_7212B0
// Address: 0x7212b0
//
_BOOL8 __fastcall sub_7212B0(__int64 a1)
{
  const char *v1; // rax
  struct stat stat_buf; // [rsp+0h] [rbp-90h] BYREF

  v1 = (const char *)sub_7212A0(a1);
  return __xstat(1, v1, &stat_buf) == 0;
}
