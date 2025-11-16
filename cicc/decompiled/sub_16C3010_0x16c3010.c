// Function: sub_16C3010
// Address: 0x16c3010
//
int __fastcall sub_16C3010(pthread_mutex_t **a1, unsigned __int8 a2)
{
  pthread_mutex_t *v2; // rbx
  int result; // eax
  pthread_mutexattr_t attr; // [rsp+Ch] [rbp-24h] BYREF

  *a1 = 0;
  v2 = (pthread_mutex_t *)malloc(0x28u);
  if ( !v2 )
    sub_16BD1C0("Allocation failed", 1u);
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, a2);
  pthread_mutex_init(v2, &attr);
  result = pthread_mutexattr_destroy(&attr);
  *a1 = v2;
  return result;
}
