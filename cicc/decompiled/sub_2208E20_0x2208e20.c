// Function: sub_2208E20
// Address: 0x2208e20
//
volatile signed __int32 *__fastcall sub_2208E20(volatile signed __int32 **a1, volatile signed __int32 **a2)
{
  volatile signed __int32 *result; // rax

  result = *a2;
  *a1 = *a2;
  if ( result != (volatile signed __int32 *)unk_4FD4F58 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd(result, 1u);
    else
      ++*result;
  }
  return result;
}
