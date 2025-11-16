// Function: sub_C8C6A0
// Address: 0xc8c6a0
//
void sub_C8C6A0()
{
  __int64 v0; // r13
  __int64 v1; // rbx
  const char *v2; // r12
  struct stat stat_buf; // [rsp+0h] [rbp-B0h] BYREF

  v0 = _InterlockedExchange64(&qword_4F84BA8, 0);
  if ( v0 )
  {
    v1 = v0;
    do
    {
      while ( 1 )
      {
        v2 = (const char *)_InterlockedExchange64((volatile __int64 *)v1, 0);
        if ( v2 )
          break;
        v1 = *(_QWORD *)(v1 + 8);
        if ( !v1 )
          goto LABEL_9;
      }
      if ( !__xstat(1, v2, &stat_buf) && (stat_buf.st_mode & 0xF000) == 0x8000 )
        unlink(v2);
      _InterlockedExchange64((volatile __int64 *)v1, (__int64)v2);
      v1 = *(_QWORD *)(v1 + 8);
    }
    while ( v1 );
  }
LABEL_9:
  _InterlockedExchange64(&qword_4F84BA8, v0);
}
