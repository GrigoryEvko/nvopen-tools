// Function: sub_3546D90
// Address: 0x3546d90
//
void __fastcall sub_3546D90(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  volatile signed __int32 *v4; // r13
  signed __int32 v5; // eax
  volatile signed __int32 *v6; // r13
  signed __int32 v7; // eax
  signed __int32 v8; // eax
  signed __int32 v9; // eax

  if ( a2 != a1 )
  {
    v2 = a2;
    do
    {
      v3 = *(_QWORD *)(v2 - 8);
      v2 -= 8;
      if ( v3 )
      {
        v4 = *(volatile signed __int32 **)(v3 + 32);
        if ( v4 )
        {
          if ( &_pthread_key_create )
          {
            v5 = _InterlockedExchangeAdd(v4 + 2, 0xFFFFFFFF);
          }
          else
          {
            v5 = *((_DWORD *)v4 + 2);
            *((_DWORD *)v4 + 2) = v5 - 1;
          }
          if ( v5 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 16LL))(v4);
            if ( &_pthread_key_create )
            {
              v8 = _InterlockedExchangeAdd(v4 + 3, 0xFFFFFFFF);
            }
            else
            {
              v8 = *((_DWORD *)v4 + 3);
              *((_DWORD *)v4 + 3) = v8 - 1;
            }
            if ( v8 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 24LL))(v4);
          }
        }
        v6 = *(volatile signed __int32 **)(v3 + 16);
        if ( v6 )
        {
          if ( &_pthread_key_create )
          {
            v7 = _InterlockedExchangeAdd(v6 + 2, 0xFFFFFFFF);
          }
          else
          {
            v7 = *((_DWORD *)v6 + 2);
            *((_DWORD *)v6 + 2) = v7 - 1;
          }
          if ( v7 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v6 + 16LL))(v6);
            if ( &_pthread_key_create )
            {
              v9 = _InterlockedExchangeAdd(v6 + 3, 0xFFFFFFFF);
            }
            else
            {
              v9 = *((_DWORD *)v6 + 3);
              *((_DWORD *)v6 + 3) = v9 - 1;
            }
            if ( v9 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v6 + 24LL))(v6);
          }
        }
        j_j___libc_free_0(v3);
      }
    }
    while ( a1 != v2 );
  }
}
