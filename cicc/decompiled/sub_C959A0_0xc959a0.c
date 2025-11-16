// Function: sub_C959A0
// Address: 0xc959a0
//
unsigned int __fastcall sub_C959A0(pthread_t a1)
{
  unsigned int result; // eax

  result = pthread_join(a1, 0);
  if ( result )
    sub_C94E30("pthread_join failed", result);
  return result;
}
