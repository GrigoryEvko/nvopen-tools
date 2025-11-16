// Function: sub_1319830
// Address: 0x1319830
//
__int64 sub_1319830()
{
  unsigned int v0; // r12d
  int v2; // eax
  sigset_t set; // [rsp+0h] [rbp-120h] BYREF
  __sigset_t oldmask; // [rsp+80h] [rbp-A0h] BYREF

  sigfillset(&set);
  v0 = pthread_sigmask(2, &set, &oldmask);
  if ( !v0 )
  {
    v0 = sub_1319820();
    v2 = pthread_sigmask(2, &oldmask, 0);
    if ( v2 )
    {
      sub_130ACF0(
        "<jemalloc>: background thread creation failed (%d), and signal mask restoration failed (%d)\n",
        v0,
        v2);
      if ( byte_4F969A5[0] )
        abort();
    }
  }
  return v0;
}
