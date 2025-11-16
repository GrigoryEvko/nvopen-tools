// Function: sub_16C6980
// Address: 0x16c6980
//
__int64 __fastcall sub_16C6980(int fd, __int64 a2)
{
  int v2; // ebx
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 result; // rax
  unsigned int v7; // ebx
  sigset_t set; // [rsp+0h] [rbp-130h] BYREF
  __sigset_t oldmask; // [rsp+80h] [rbp-B0h] BYREF

  v2 = sigfillset(&set);
  sub_2241E50(&set, a2, v3, v4, v5);
  if ( v2 < 0 )
    return (unsigned int)*__errno_location();
  LODWORD(result) = pthread_sigmask(2, &set, &oldmask);
  if ( (_DWORD)result )
    return (unsigned int)result;
  if ( close(fd) >= 0 )
    return (unsigned int)pthread_sigmask(2, &oldmask, 0);
  v7 = *__errno_location();
  LODWORD(result) = pthread_sigmask(2, &oldmask, 0);
  if ( !v7 )
    return (unsigned int)result;
  return v7;
}
