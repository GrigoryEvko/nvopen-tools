// Function: sub_2215790
// Address: 0x2215790
//
volatile signed __int32 *__fastcall sub_2215790(volatile signed __int32 **a1)
{
  volatile signed __int32 *result; // rax

  result = (volatile signed __int32 *)*((unsigned int *)*a1 - 2);
  if ( (int)result >= 0 )
    return sub_2215730(a1);
  return result;
}
