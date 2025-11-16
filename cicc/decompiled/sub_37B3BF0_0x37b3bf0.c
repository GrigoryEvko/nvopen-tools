// Function: sub_37B3BF0
// Address: 0x37b3bf0
//
void __fastcall sub_37B3BF0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // r12
  volatile signed __int32 *v4; // r13
  signed __int32 v5; // eax
  signed __int32 v6; // eax
  volatile signed __int32 *v7; // r13
  signed __int32 v8; // eax
  signed __int32 v9; // eax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi

  *a1 = &unk_4A3D478;
  v2 = a1[21];
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = a1[20];
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
    v7 = *(volatile signed __int32 **)(v3 + 16);
    if ( v7 )
    {
      if ( &_pthread_key_create )
      {
        v8 = _InterlockedExchangeAdd(v7 + 2, 0xFFFFFFFF);
      }
      else
      {
        v8 = *((_DWORD *)v7 + 2);
        *((_DWORD *)v7 + 2) = v8 - 1;
      }
      if ( v8 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v7 + 16LL))(v7);
        if ( &_pthread_key_create )
        {
          v9 = _InterlockedExchangeAdd(v7 + 3, 0xFFFFFFFF);
        }
        else
        {
          v9 = *((_DWORD *)v7 + 3);
          *((_DWORD *)v7 + 3) = v9 - 1;
        }
        if ( v9 == 1 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v7 + 24LL))(v7);
      }
    }
    j_j___libc_free_0(v3);
  }
  v10 = a1[12];
  if ( v10 )
    j_j___libc_free_0(v10);
  v11 = a1[9];
  if ( v11 )
    j_j___libc_free_0(v11);
  v12 = a1[6];
  if ( v12 )
    j_j___libc_free_0(v12);
  v13 = a1[3];
  if ( v13 )
    j_j___libc_free_0(v13);
}
