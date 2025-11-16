// Function: sub_2207640
// Address: 0x2207640
//
__int64 __fastcall sub_2207640(__int64 a1)
{
  __int64 result; // rax

  if ( &_pthread_key_create )
  {
    result = (unsigned int)_InterlockedExchange((volatile __int32 *)a1, 1);
    if ( (result & 0x10000) != 0 )
      return syscall(202, a1, 1, 0x7FFFFFFF);
  }
  else
  {
    *(_BYTE *)(a1 + 1) = 0;
    *(_BYTE *)a1 = 1;
  }
  return result;
}
