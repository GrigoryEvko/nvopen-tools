// Function: sub_166D520
// Address: 0x166d520
//
__int64 __fastcall sub_166D520(_QWORD *a1)
{
  __int64 result; // rax
  volatile signed __int32 *v2; // r12
  volatile signed __int32 *v3; // r13

  result = a1[20];
  v2 = (volatile signed __int32 *)a1[24];
  a1[23] = *(_QWORD *)result;
  v3 = *(volatile signed __int32 **)(result + 8);
  if ( v3 != v2 )
  {
    if ( v3 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(v3 + 2, 1u);
      else
        ++*((_DWORD *)v3 + 2);
      v2 = (volatile signed __int32 *)a1[24];
    }
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
          result = (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 24LL))(v2);
      }
    }
    a1[24] = v3;
  }
  return result;
}
