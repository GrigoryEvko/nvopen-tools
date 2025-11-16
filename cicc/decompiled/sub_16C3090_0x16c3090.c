// Function: sub_16C3090
// Address: 0x16c3090
//
void __fastcall sub_16C3090(pthread_mutex_t **a1)
{
  pthread_mutex_t *v1; // r12

  v1 = *a1;
  pthread_mutex_destroy(*a1);
  _libc_free((unsigned __int64)v1);
}
