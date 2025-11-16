// Function: sub_166D440
// Address: 0x166d440
//
__int64 __fastcall sub_166D440(_QWORD *a1)
{
  __int64 *v1; // r13
  __int64 result; // rax
  volatile signed __int32 *v3; // r12
  volatile signed __int32 *v4; // rbx

  v1 = (__int64 *)a1[20];
  result = a1[23];
  v3 = (volatile signed __int32 *)v1[1];
  *v1 = result;
  v4 = (volatile signed __int32 *)a1[24];
  if ( v4 != v3 )
  {
    if ( v4 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(v4 + 2, 1u);
      else
        ++*((_DWORD *)v4 + 2);
      v3 = (volatile signed __int32 *)v1[1];
    }
    if ( v3 )
    {
      if ( &_pthread_key_create )
      {
        result = (unsigned int)_InterlockedExchangeAdd(v3 + 2, 0xFFFFFFFF);
      }
      else
      {
        result = *((unsigned int *)v3 + 2);
        *((_DWORD *)v3 + 2) = result - 1;
      }
      if ( (_DWORD)result == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v3 + 16LL))(v3);
        if ( &_pthread_key_create )
        {
          result = (unsigned int)_InterlockedExchangeAdd(v3 + 3, 0xFFFFFFFF);
        }
        else
        {
          result = *((unsigned int *)v3 + 3);
          *((_DWORD *)v3 + 3) = result - 1;
        }
        if ( (_DWORD)result == 1 )
          result = (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v3 + 24LL))(v3);
      }
    }
    v1[1] = (__int64)v4;
  }
  return result;
}
