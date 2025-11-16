// Function: sub_16C9010
// Address: 0x16c9010
//
void __fastcall sub_16C9010(pthread_rwlock_t **a1)
{
  pthread_rwlock_t *v1; // r12

  v1 = *a1;
  pthread_rwlock_destroy(*a1);
  _libc_free((unsigned __int64)v1);
}
