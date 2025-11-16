// Function: sub_16CC630
// Address: 0x16cc630
//
int __fastcall sub_16CC630(unsigned int sig)
{
  struct sigaction *v1; // rbx
  struct sigaction *v2; // r12
  __int64 v3; // r14
  __int64 i; // rbx
  const char *v5; // r12
  __int64 v7; // rax
  __int64 (*v8)(void); // rax
  sigset_t set; // [rsp+0h] [rbp-140h] BYREF
  struct stat stat_buf; // [rsp+80h] [rbp-C0h] BYREF

  if ( dword_4FA1080 )
  {
    v1 = &stru_4FA0680;
    v2 = (struct sigaction *)((char *)&unk_4FA0720 + 160 * (unsigned int)(dword_4FA1080 - 1));
    do
    {
      sigaction((int)v1[1].sa_handler, v1, 0);
      _InterlockedSub(&dword_4FA1080, 1u);
      v1 = (struct sigaction *)((char *)v1 + 160);
    }
    while ( v1 != v2 );
  }
  sigfillset(&set);
  sigprocmask(1, &set, 0);
  v3 = _InterlockedExchange64(&qword_4FA1088, 0);
  for ( i = v3; i; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      v5 = (const char *)_InterlockedExchange64((volatile __int64 *)i, 0);
      if ( v5 )
      {
        if ( !__xstat(1, v5, &stat_buf) && (stat_buf.st_mode & 0xF000) == 0x8000 )
          break;
      }
      i = *(_QWORD *)(i + 8);
      if ( !i )
        goto LABEL_11;
    }
    unlink(v5);
    _InterlockedExchange64((volatile __int64 *)i, (__int64)v5);
  }
LABEL_11:
  _InterlockedExchange64(&qword_4FA1088, v3);
  if ( sig > 0xF )
    return sub_16CC5C0();
  v7 = 46086;
  if ( !_bittest64(&v7, sig) )
    return sub_16CC5C0();
  v8 = (__int64 (*)(void))_InterlockedExchange64(&qword_4FA1090, 0);
  if ( v8 )
    return v8();
  else
    return raise(sig);
}
