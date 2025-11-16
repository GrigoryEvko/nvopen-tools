// Function: sub_7217C0
// Address: 0x7217c0
//
int __fastcall sub_7217C0(char *filename, __dev_t *a2)
{
  int result; // eax
  struct stat stat_buf; // [rsp+0h] [rbp-A0h] BYREF

  sub_7217B0(a2);
  result = __xstat(1, filename, &stat_buf);
  if ( !result )
  {
    *a2 = stat_buf.st_dev;
    result = stat_buf.st_ino;
    a2[1] = stat_buf.st_ino;
  }
  return result;
}
