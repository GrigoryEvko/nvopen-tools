// Function: sub_A191D0
// Address: 0xa191d0
//
__int64 __fastcall sub_A191D0(volatile signed __int32 *a1)
{
  __int64 result; // rax

  if ( &_pthread_key_create )
  {
    result = (unsigned int)_InterlockedExchangeAdd(a1 + 2, 0xFFFFFFFF);
  }
  else
  {
    result = *((unsigned int *)a1 + 2);
    *((_DWORD *)a1 + 2) = result - 1;
  }
  if ( (_DWORD)result == 1 )
  {
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)a1 + 16LL))(a1);
    if ( &_pthread_key_create )
    {
      result = (unsigned int)_InterlockedExchangeAdd(a1 + 3, 0xFFFFFFFF);
    }
    else
    {
      result = *((unsigned int *)a1 + 3);
      *((_DWORD *)a1 + 3) = result - 1;
    }
    if ( (_DWORD)result == 1 )
      return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)a1 + 24LL))(a1);
  }
  return result;
}
