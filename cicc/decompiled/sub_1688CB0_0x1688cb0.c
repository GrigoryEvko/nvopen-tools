// Function: sub_1688CB0
// Address: 0x1688cb0
//
__int64 sub_1688CB0()
{
  pthread_mutexattr_t attr; // [rsp+Ch] [rbp-14h] BYREF

  if ( !unk_4F9F820 )
  {
    pthread_key_create(&dword_4F9F868, sub_1688F30);
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, 1);
    pthread_mutex_init(&stru_4F9F840, &attr);
    pthread_mutexattr_destroy(&attr);
    dword_4F9F874 = sched_get_priority_max(2);
    dword_4F9F870 = sched_get_priority_min(2);
    dword_4F9F86C = dword_4F9F874 - dword_4F9F870 + 1;
    unk_4F9F708 = &unk_4F9F720;
    unk_4F9F820 = &unk_4F9F600;
    unk_4F9F5E0 = &unk_4F9F3C0;
    unk_4F9F4C8 = &unk_4F9F4E0;
  }
  return 1;
}
