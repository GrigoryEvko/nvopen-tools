// Function: sub_22090A0
// Address: 0x22090a0
//
volatile signed __int32 **__fastcall sub_22090A0(volatile signed __int32 **a1, volatile signed __int32 **a2)
{
  volatile signed __int32 **result; // rax
  volatile signed __int32 *v3; // rcx
  volatile signed __int32 *v4; // rdx
  volatile signed __int32 *v5; // rbp
  volatile signed __int32 v6; // edx

  result = a1;
  v3 = *a2;
  v4 = (volatile signed __int32 *)unk_4FD4F58;
  if ( *a2 != (volatile signed __int32 *)unk_4FD4F58 )
  {
    if ( &_pthread_key_create )
    {
      _InterlockedAdd(v3, 1u);
      v4 = (volatile signed __int32 *)unk_4FD4F58;
    }
    else
    {
      ++*v3;
    }
  }
  v5 = *a1;
  if ( *a1 == v4 )
    goto LABEL_7;
  if ( &_pthread_key_create )
  {
    if ( _InterlockedExchangeAdd(v5, 0xFFFFFFFF) != 1 )
      goto LABEL_7;
  }
  else
  {
    v6 = (*v5)--;
    if ( v6 != 1 )
    {
LABEL_7:
      *a1 = *a2;
      return result;
    }
  }
  sub_2208EC0(v5);
  j___libc_free_0((unsigned __int64)v5);
  *a1 = *a2;
  return a1;
}
