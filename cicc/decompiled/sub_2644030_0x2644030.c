// Function: sub_2644030
// Address: 0x2644030
//
void __fastcall sub_2644030(unsigned __int64 *a1)
{
  unsigned __int64 v1; // r14
  unsigned __int64 v2; // r12
  volatile signed __int32 *v3; // r13
  signed __int32 v4; // eax
  signed __int32 v5; // eax

  v1 = a1[1];
  v2 = *a1;
  if ( v1 != *a1 )
  {
    do
    {
      while ( 1 )
      {
        v3 = *(volatile signed __int32 **)(v2 + 8);
        if ( v3 )
        {
          if ( &_pthread_key_create )
          {
            v4 = _InterlockedExchangeAdd(v3 + 2, 0xFFFFFFFF);
          }
          else
          {
            v4 = *((_DWORD *)v3 + 2);
            *((_DWORD *)v3 + 2) = v4 - 1;
          }
          if ( v4 == 1 )
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
        v2 += 16LL;
        if ( v1 == v2 )
          goto LABEL_12;
      }
      v2 += 16LL;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v3 + 24LL))(v3);
    }
    while ( v1 != v2 );
LABEL_12:
    v2 = *a1;
  }
  if ( v2 )
    j_j___libc_free_0(v2);
}
