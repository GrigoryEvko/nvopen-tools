// Function: sub_2D9FFC0
// Address: 0x2d9ffc0
//
__int64 __fastcall sub_2D9FFC0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 result; // rax
  volatile signed __int32 *v4; // r12

  v2 = *a2;
  result = a2[1];
  a2[1] = 0;
  *a2 = 0;
  v4 = (volatile signed __int32 *)a1[1];
  *a1 = v2;
  a1[1] = result;
  if ( v4 )
  {
    if ( &_pthread_key_create )
    {
      result = (unsigned int)_InterlockedExchangeAdd(v4 + 2, 0xFFFFFFFF);
    }
    else
    {
      result = *((unsigned int *)v4 + 2);
      *((_DWORD *)v4 + 2) = result - 1;
    }
    if ( (_DWORD)result == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 16LL))(v4);
      if ( &_pthread_key_create )
      {
        result = (unsigned int)_InterlockedExchangeAdd(v4 + 3, 0xFFFFFFFF);
      }
      else
      {
        result = *((unsigned int *)v4 + 3);
        *((_DWORD *)v4 + 3) = result - 1;
      }
      if ( (_DWORD)result == 1 )
        return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 24LL))(v4);
    }
  }
  return result;
}
