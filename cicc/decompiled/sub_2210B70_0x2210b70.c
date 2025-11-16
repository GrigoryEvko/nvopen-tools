// Function: sub_2210B70
// Address: 0x2210b70
//
unsigned int __fastcall sub_2210B70(pthread_cond_t *a1)
{
  unsigned int result; // eax

  result = pthread_cond_broadcast(a1);
  if ( result )
    sub_4264C5(result);
  return result;
}
