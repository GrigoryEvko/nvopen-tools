// Function: sub_16C8FC0
// Address: 0x16c8fc0
//
int __fastcall sub_16C8FC0(pthread_rwlock_t **a1)
{
  pthread_rwlock_t *v1; // rbx
  int result; // eax

  *a1 = 0;
  v1 = (pthread_rwlock_t *)malloc(0x38u);
  if ( !v1 )
    sub_16BD1C0("Allocation failed", 1u);
  result = pthread_rwlock_init(v1, 0);
  *a1 = v1;
  return result;
}
