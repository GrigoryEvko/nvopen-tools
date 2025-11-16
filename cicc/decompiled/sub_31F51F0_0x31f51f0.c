// Function: sub_31F51F0
// Address: 0x31f51f0
//
__int64 __fastcall sub_31F51F0(_QWORD *a1)
{
  __int64 result; // rax
  volatile signed __int32 *v2; // r12

  result = (__int64)&unk_4A352E0;
  v2 = (volatile signed __int32 *)a1[2];
  *a1 = &unk_4A352E0;
  if ( v2 )
  {
    if ( &_pthread_key_create )
    {
      result = (unsigned int)_InterlockedExchangeAdd(v2 + 2, 0xFFFFFFFF);
    }
    else
    {
      result = *((unsigned int *)v2 + 2);
      *((_DWORD *)v2 + 2) = result - 1;
    }
    if ( (_DWORD)result == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 16LL))(v2);
      if ( &_pthread_key_create )
      {
        result = (unsigned int)_InterlockedExchangeAdd(v2 + 3, 0xFFFFFFFF);
      }
      else
      {
        result = *((unsigned int *)v2 + 3);
        *((_DWORD *)v2 + 3) = result - 1;
      }
      if ( (_DWORD)result == 1 )
        return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 24LL))(v2);
    }
  }
  return result;
}
