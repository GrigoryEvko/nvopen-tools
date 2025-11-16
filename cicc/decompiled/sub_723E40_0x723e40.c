// Function: sub_723E40
// Address: 0x723e40
//
__int64 __fastcall sub_723E40(__int64 a1, __time_t *a2)
{
  const char *v2; // rax
  __int64 result; // rax
  struct stat stat_buf; // [rsp+0h] [rbp-A0h] BYREF

  v2 = (const char *)sub_7212A0(a1);
  if ( __xstat(1, v2, &stat_buf) )
  {
    result = 0;
    if ( a2 )
      *a2 = 0;
  }
  else
  {
    LOBYTE(result) = (stat_buf.st_mode & 0xF000) == 0x8000;
    if ( a2 && (stat_buf.st_mode & 0xF000) == 0x8000 )
    {
      *a2 = stat_buf.st_mtim.tv_sec;
      return 1;
    }
    else
    {
      return (unsigned __int8)result;
    }
  }
  return result;
}
