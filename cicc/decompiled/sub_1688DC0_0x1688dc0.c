// Function: sub_1688DC0
// Address: 0x1688dc0
//
bool __fastcall sub_1688DC0(pthread_mutex_t *mutex)
{
  int v1; // ebx
  pthread_mutexattr_t attr; // [rsp+Ch] [rbp-24h] BYREF

  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, 1);
  v1 = pthread_mutex_init(mutex, &attr);
  pthread_mutexattr_destroy(&attr);
  return v1 == 0;
}
