// Function: sub_2210B30
// Address: 0x2210b30
//
int __fastcall sub_2210B30(pthread_cond_t *a1, pthread_mutex_t **a2)
{
  int result; // eax

  result = pthread_cond_wait(a1, *a2);
  if ( result )
    sub_2207530();
  return result;
}
