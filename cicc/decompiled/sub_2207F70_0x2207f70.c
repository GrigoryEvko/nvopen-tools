// Function: sub_2207F70
// Address: 0x2207f70
//
__off_t __fastcall sub_2207F70(FILE **a1)
{
  int v1; // eax
  __off_t result; // rax
  int v3; // eax
  __off_t st_size; // rbx
  int v5; // eax
  int v6; // [rsp+4h] [rbp-B4h] BYREF
  struct pollfd fds; // [rsp+8h] [rbp-B0h] BYREF
  struct stat64 stat_buf; // [rsp+10h] [rbp-A8h] BYREF

  v6 = 0;
  v1 = sub_2207D30(a1);
  if ( ioctl(v1, 0x541Bu, &v6) || (result = v6, v6 < 0) )
  {
    fds.fd = sub_2207D30(a1);
    fds.events = 1;
    if ( poll(&fds, 1u, 0) > 0
      && (v3 = sub_2207D30(a1), !__fxstat64(1, v3, &stat_buf))
      && (stat_buf.st_mode & 0xF000) == 0x8000 )
    {
      st_size = stat_buf.st_size;
      v5 = sub_2207D30(a1);
      return st_size - lseek64(v5, 0, 1);
    }
    else
    {
      return 0;
    }
  }
  return result;
}
