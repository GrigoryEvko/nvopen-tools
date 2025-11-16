// Function: sub_C8BD80
// Address: 0xc8bd80
//
int sub_C8BD80()
{
  unsigned int v0; // eax
  int result; // eax
  int v2; // r12d
  char *i; // r13
  __int64 v4; // rbx
  struct sigaction *v5; // rbx
  int v6; // r12d
  int *j; // r13
  __int64 v8; // rbx
  struct sigaction *v9; // rbx
  __int64 v10; // rbx
  struct sigaction *v11; // rbx
  __int64 v12; // rbx
  pthread_mutex_t *mutex; // [rsp+8h] [rbp-D8h]
  struct sigaction act; // [rsp+10h] [rbp-D0h] BYREF

  if ( !qword_4F84170 )
    sub_C7D570((__int64 *)&qword_4F84170, (__int64 (*)(void))sub_BC3580, (__int64)sub_BC3540);
  mutex = qword_4F84170;
  if ( &_pthread_key_create )
  {
    v0 = pthread_mutex_lock(qword_4F84170);
    if ( v0 )
      sub_4264C5(v0);
  }
  result = dword_4F84BA0;
  if ( !dword_4F84BA0 )
  {
    v2 = 1;
    for ( i = (char *)&unk_3F67570; ; v2 = *(_DWORD *)i )
    {
      v4 = 160LL * (unsigned int)dword_4F84BA0;
      act.sa_handler = (__sighandler_t)sub_C8C540;
      act.sa_flags = -939524092;
      sigemptyset(&act.sa_mask);
      v5 = (struct sigaction *)((char *)&::act + v4);
      sigaction(v2, &act, v5);
      LODWORD(v5[1].sa_handler) = v2;
      _InterlockedAdd(&dword_4F84BA0, 1u);
      i += 4;
      if ( i == "SmallVector capacity unable to grow. Already at maximum size " )
        break;
    }
    v6 = 4;
    for ( j = (int *)&unk_3F67540; ; v6 = *j )
    {
      v8 = 160LL * (unsigned int)dword_4F84BA0;
      act.sa_handler = (__sighandler_t)sub_C8C540;
      act.sa_flags = -939524092;
      sigemptyset(&act.sa_mask);
      v9 = (struct sigaction *)((char *)&::act + v8);
      sigaction(v6, &act, v9);
      LODWORD(v9[1].sa_handler) = v6;
      _InterlockedAdd(&dword_4F84BA0, 1u);
      if ( ++j == (int *)&unk_3F67568 )
        break;
    }
    if ( qword_4F84BC8 )
    {
      v10 = 160LL * (unsigned int)dword_4F84BA0;
      act.sa_handler = (__sighandler_t)sub_C8C540;
      act.sa_flags = -939524092;
      sigemptyset(&act.sa_mask);
      v11 = (struct sigaction *)((char *)&::act + v10);
      sigaction(13, &act, v11);
      LODWORD(v11[1].sa_handler) = 13;
      _InterlockedAdd(&dword_4F84BA0, 1u);
    }
    v12 = (unsigned int)dword_4F84BA0;
    act.sa_handler = (__sighandler_t)sub_C8B430;
    act.sa_flags = 0x8000000;
    sigemptyset(&act.sa_mask);
    result = sigaction(10, &act, (struct sigaction *)((char *)&::act + 160 * v12));
    *((_DWORD *)&::act + 40 * v12 + 38) = 10;
    _InterlockedAdd(&dword_4F84BA0, 1u);
  }
  if ( &_pthread_key_create )
    return pthread_mutex_unlock(mutex);
  return result;
}
