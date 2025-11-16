// Function: sub_CA4380
// Address: 0xca4380
//
__int64 __fastcall sub_CA4380(__int64 a1)
{
  volatile signed __int32 *v1; // r12
  signed __int32 v2; // eax
  signed __int32 v4; // eax

  v1 = *(volatile signed __int32 **)(a1 + 72);
  *(_QWORD *)(a1 + 16) = off_4979CB0;
  if ( v1 )
  {
    if ( &_pthread_key_create )
    {
      v2 = _InterlockedExchangeAdd(v1 + 2, 0xFFFFFFFF);
    }
    else
    {
      v2 = *((_DWORD *)v1 + 2);
      *((_DWORD *)v1 + 2) = v2 - 1;
    }
    if ( v2 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v1 + 16LL))(v1);
      if ( &_pthread_key_create )
      {
        v4 = _InterlockedExchangeAdd(v1 + 3, 0xFFFFFFFF);
      }
      else
      {
        v4 = *((_DWORD *)v1 + 3);
        *((_DWORD *)v1 + 3) = v4 - 1;
      }
      if ( v4 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v1 + 24LL))(v1);
    }
  }
  return sub_CA4290((_QWORD *)(a1 + 16));
}
