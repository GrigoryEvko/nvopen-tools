// Function: sub_264F9A0
// Address: 0x264f9a0
//
__int64 *__fastcall sub_264F9A0(__int64 a1)
{
  __int64 *result; // rax
  signed __int32 v3; // eax
  volatile signed __int32 *v4; // r12
  __int64 v5; // rdi
  signed __int32 v6; // eax
  __int64 *v7; // [rsp+8h] [rbp-28h] BYREF

  result = *(__int64 **)(a1 + 48);
  v7 = result;
  while ( *(__int64 **)(a1 + 56) != result )
  {
    v4 = (volatile signed __int32 *)result[1];
    v5 = *result;
    if ( v4 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(v4 + 2, 1u);
      else
        ++*((_DWORD *)v4 + 2);
    }
    if ( *(_BYTE *)(v5 + 16) )
      v7 += 2;
    else
      sub_264E780(v5, (__int64 *)&v7, 1);
    if ( v4 )
    {
      if ( &_pthread_key_create )
      {
        v3 = _InterlockedExchangeAdd(v4 + 2, 0xFFFFFFFF);
      }
      else
      {
        v3 = *((_DWORD *)v4 + 2);
        *((_DWORD *)v4 + 2) = v3 - 1;
      }
      if ( v3 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 16LL))(v4);
        if ( &_pthread_key_create )
        {
          v6 = _InterlockedExchangeAdd(v4 + 3, 0xFFFFFFFF);
        }
        else
        {
          v6 = *((_DWORD *)v4 + 3);
          *((_DWORD *)v4 + 3) = v6 - 1;
        }
        if ( v6 == 1 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 24LL))(v4);
      }
    }
    result = v7;
  }
  return result;
}
