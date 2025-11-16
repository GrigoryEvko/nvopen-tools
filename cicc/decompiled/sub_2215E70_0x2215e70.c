// Function: sub_2215E70
// Address: 0x2215e70
//
volatile signed __int32 *__fastcall sub_2215E70(volatile signed __int32 **a1, volatile signed __int32 **a2)
{
  volatile signed __int32 *result; // rax
  volatile signed __int32 *v4; // rdi
  char v5[9]; // [rsp+Fh] [rbp-9h] BYREF

  result = *a2;
  v4 = *a2 - 6;
  if ( *((int *)*a2 - 2) < 0 )
  {
    result = (volatile signed __int32 *)sub_2215A20((__int64)v4, (__int64)v5, 0);
    *a1 = result;
  }
  else
  {
    if ( v4 != (volatile signed __int32 *)&unk_4FD67C0 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(result - 2, 1u);
      else
        ++*((_DWORD *)result - 2);
    }
    *a1 = result;
  }
  return result;
}
