// Function: sub_2215730
// Address: 0x2215730
//
volatile signed __int32 *__fastcall sub_2215730(volatile signed __int32 **a1)
{
  volatile signed __int32 *result; // rax

  result = *a1;
  if ( *a1 - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
  {
    if ( *((int *)result - 2) > 0 )
      sub_2215540(a1, 0, 0, 0);
    result = *a1;
    *((_DWORD *)*a1 - 2) = -1;
  }
  return result;
}
