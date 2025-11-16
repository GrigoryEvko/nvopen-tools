// Function: sub_166D2C0
// Address: 0x166d2c0
//
void __fastcall sub_166D2C0(_QWORD *a1)
{
  volatile signed __int32 *v2; // r12
  signed __int32 v3; // eax
  unsigned __int64 v4; // rdi
  signed __int32 v5; // eax

  v2 = (volatile signed __int32 *)a1[24];
  *a1 = &off_49EE3A0;
  if ( v2 )
  {
    if ( &_pthread_key_create )
    {
      v3 = _InterlockedExchangeAdd(v2 + 2, 0xFFFFFFFF);
    }
    else
    {
      v3 = *((_DWORD *)v2 + 2);
      *((_DWORD *)v2 + 2) = v3 - 1;
    }
    if ( v3 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 16LL))(v2);
      if ( &_pthread_key_create )
      {
        v5 = _InterlockedExchangeAdd(v2 + 3, 0xFFFFFFFF);
      }
      else
      {
        v5 = *((_DWORD *)v2 + 3);
        *((_DWORD *)v2 + 3) = v5 - 1;
      }
      if ( v5 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 24LL))(v2);
    }
  }
  v4 = a1[12];
  if ( v4 != a1[11] )
    _libc_free(v4);
}
