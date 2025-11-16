// Function: sub_2647930
// Address: 0x2647930
//
void __fastcall sub_2647930(__int64 a1)
{
  __int64 *v1; // r13
  signed __int32 v2; // eax
  volatile signed __int32 *v3; // r12
  __int64 v4; // rsi
  signed __int32 v5; // eax

  v1 = *(__int64 **)(a1 + 72);
  while ( *(__int64 **)(a1 + 80) != v1 )
  {
    while ( 1 )
    {
      v3 = (volatile signed __int32 *)v1[1];
      v4 = *v1;
      if ( v3 )
      {
        if ( &_pthread_key_create )
          _InterlockedAdd(v3 + 2, 1u);
        else
          ++*((_DWORD *)v3 + 2);
      }
      if ( *(_BYTE *)(v4 + 16) )
      {
        v1 += 2;
      }
      else
      {
        sub_2647840(*(_QWORD *)(v4 + 8), v4);
        v1 = (__int64 *)sub_26476C0(a1 + 72, (__int64)v1);
      }
      if ( v3 )
      {
        if ( &_pthread_key_create )
        {
          v2 = _InterlockedExchangeAdd(v3 + 2, 0xFFFFFFFF);
        }
        else
        {
          v2 = *((_DWORD *)v3 + 2);
          *((_DWORD *)v3 + 2) = v2 - 1;
        }
        if ( v2 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v3 + 16LL))(v3);
          if ( &_pthread_key_create )
          {
            v5 = _InterlockedExchangeAdd(v3 + 3, 0xFFFFFFFF);
          }
          else
          {
            v5 = *((_DWORD *)v3 + 3);
            *((_DWORD *)v3 + 3) = v5 - 1;
          }
          if ( v5 == 1 )
            break;
        }
      }
      if ( *(__int64 **)(a1 + 80) == v1 )
        return;
    }
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v3 + 24LL))(v3);
  }
}
