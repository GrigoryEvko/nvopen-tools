// Function: sub_2212A20
// Address: 0x2212a20
//
__int64 __fastcall sub_2212A20(volatile signed __int32 *a1)
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
    return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)a1 + 8LL))(a1);
  return result;
}
