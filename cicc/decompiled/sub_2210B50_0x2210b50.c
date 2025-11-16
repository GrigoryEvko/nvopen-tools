// Function: sub_2210B50
// Address: 0x2210b50
//
unsigned int __fastcall sub_2210B50(pthread_cond_t *a1)
{
  unsigned int result; // eax

  result = pthread_cond_signal(a1);
  if ( result )
    sub_4264C5(result);
  return result;
}
