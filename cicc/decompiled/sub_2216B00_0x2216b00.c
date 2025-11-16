// Function: sub_2216B00
// Address: 0x2216b00
//
__int64 __fastcall sub_2216B00(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  void *v4; // rdi
  _BYTE v5[9]; // [rsp+Fh] [rbp-9h] BYREF

  result = *a2;
  v4 = (void *)(*a2 - 24);
  if ( *(int *)(*a2 - 8) < 0 )
  {
    result = sub_22166A0((__int64)v4, (__int64)v5, 0);
    *a1 = result;
  }
  else
  {
    if ( v4 != &unk_4FD67E0 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd((volatile signed __int32 *)(result - 8), 1u);
      else
        ++*(_DWORD *)(result - 8);
    }
    *a1 = result;
  }
  return result;
}
