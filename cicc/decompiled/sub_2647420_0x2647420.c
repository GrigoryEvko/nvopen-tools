// Function: sub_2647420
// Address: 0x2647420
//
__int64 __fastcall sub_2647420(volatile signed __int32 **a1, volatile signed __int32 *a2)
{
  volatile signed __int32 *v3; // rdi
  __int64 result; // rax

  v3 = *a1;
  if ( v3 != a2 )
  {
    if ( a2 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(a2 + 2, 1u);
      else
        ++*((_DWORD *)a2 + 2);
      v3 = *a1;
    }
    if ( v3 )
      result = sub_A191D0(v3);
    *a1 = a2;
  }
  return result;
}
