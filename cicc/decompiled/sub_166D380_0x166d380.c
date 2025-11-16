// Function: sub_166D380
// Address: 0x166d380
//
__int64 __fastcall sub_166D380(_QWORD *a1)
{
  volatile signed __int32 *v2; // r13
  signed __int32 v3; // eax
  unsigned __int64 v4; // rdi
  signed __int32 v6; // eax

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
        v6 = _InterlockedExchangeAdd(v2 + 3, 0xFFFFFFFF);
      }
      else
      {
        v6 = *((_DWORD *)v2 + 3);
        *((_DWORD *)v2 + 3) = v6 - 1;
      }
      if ( v6 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 24LL))(v2);
    }
  }
  v4 = a1[12];
  if ( v4 != a1[11] )
    _libc_free(v4);
  return j_j___libc_free_0(a1, 200);
}
