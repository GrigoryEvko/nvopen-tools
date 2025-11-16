// Function: sub_1253E90
// Address: 0x1253e90
//
__int64 __fastcall sub_1253E90(_QWORD *a1)
{
  __int64 result; // rax
  volatile signed __int32 *v2; // r12
  void (*v3)(); // rax
  __int64 (__fastcall *v4)(__int64); // rdx

  result = (__int64)&unk_49E6870;
  v2 = (volatile signed __int32 *)a1[2];
  *a1 = &unk_49E6870;
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
      v3 = *(void (**)())(*(_QWORD *)v2 + 16LL);
      if ( v3 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v3)(v2);
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
      {
        v4 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 24LL);
        if ( v4 == sub_9C26E0 )
          return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 8LL))(v2);
        else
          return v4((__int64)v2);
      }
    }
  }
  return result;
}
