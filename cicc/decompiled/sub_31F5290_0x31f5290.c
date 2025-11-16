// Function: sub_31F5290
// Address: 0x31f5290
//
void __fastcall sub_31F5290(_QWORD *a1)
{
  volatile signed __int32 *v1; // r13
  signed __int32 v2; // eax
  signed __int32 v3; // eax

  v1 = (volatile signed __int32 *)a1[2];
  *a1 = &unk_4A352E0;
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
        v3 = _InterlockedExchangeAdd(v1 + 3, 0xFFFFFFFF);
      }
      else
      {
        v3 = *((_DWORD *)v1 + 3);
        *((_DWORD *)v1 + 3) = v3 - 1;
      }
      if ( v3 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v1 + 24LL))(v1);
    }
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
