// Function: sub_C8C540
// Address: 0xc8c540
//
__pid_t __fastcall sub_C8C540(int a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  const char *v4; // r12
  int v5; // r12d
  int v6; // ebx
  __pid_t result; // eax
  __int64 (*v8)(void); // rax
  __int64 (*v9)(void); // rax
  int sig; // [rsp+Ch] [rbp-144h] BYREF
  sigset_t set; // [rsp+10h] [rbp-140h] BYREF
  struct stat stat_buf; // [rsp+90h] [rbp-C0h] BYREF

  sig = a1;
  sub_C8C4D0();
  sigfillset(&set);
  sigprocmask(1, &set, 0);
  v2 = _InterlockedExchange64(&qword_4F84BA8, 0);
  if ( v2 )
  {
    v3 = v2;
    do
    {
      while ( 1 )
      {
        v4 = (const char *)_InterlockedExchange64((volatile __int64 *)v3, 0);
        if ( v4 )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          goto LABEL_9;
      }
      if ( !__xstat(1, v4, &stat_buf) && (stat_buf.st_mode & 0xF000) == 0x8000 )
        unlink(v4);
      _InterlockedExchange64((volatile __int64 *)v3, (__int64)v4);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v3 );
  }
LABEL_9:
  _InterlockedExchange64(&qword_4F84BA8, v2);
  if ( sig == 13 )
  {
    v9 = (__int64 (*)(void))_InterlockedExchange64(&qword_4F84BC8, 0);
    if ( v9 )
      return v9();
  }
  if ( "SmallVector capacity unable to grow. Already at maximum size " != (char *)sub_C8B460(
                                                                                    &aSmallvectorCap[-16],
                                                                                    (__int64)"SmallVector capacity unable"
                                                                                             " to grow. Already at maximum size ",
                                                                                    &sig) )
  {
    v8 = (__int64 (*)(void))_InterlockedExchange64(&qword_4F84BD8, 0);
    if ( v8 )
      return v8();
    v5 = sig;
    return raise(v5);
  }
  v5 = sig;
  if ( sig == 13 )
    return raise(v5);
  sub_C8C440();
  v6 = *(_DWORD *)(a2 + 16);
  result = getpid();
  if ( v6 != result )
    return raise(v5);
  return result;
}
