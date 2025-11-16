// Function: sub_721580
// Address: 0x721580
//
_BOOL8 __fastcall sub_721580(char *filename)
{
  int v1; // r8d
  _BOOL8 result; // rax
  struct stat stat_buf; // [rsp+0h] [rbp-90h] BYREF

  v1 = __xstat(1, filename, &stat_buf);
  result = 0;
  if ( !v1 )
    return (stat_buf.st_mode & 0xF000) == 0x4000;
  return result;
}
