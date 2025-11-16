// Function: sub_22420C0
// Address: 0x22420c0
//
unsigned int __fastcall sub_22420C0(pthread_t *a1, void **a2)
{
  unsigned int result; // eax

  result = pthread_create(a1, 0, start_routine, *a2);
  if ( result )
    sub_4264C5(result);
  *a2 = 0;
  return result;
}
