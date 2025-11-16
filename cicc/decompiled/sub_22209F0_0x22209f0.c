// Function: sub_22209F0
// Address: 0x22209f0
//
void __fastcall sub_22209F0(_QWORD *a1)
{
  volatile signed __int32 *v2; // rdi
  signed __int32 v3; // eax

  *a1 = off_49D28B8;
  v2 = (volatile signed __int32 *)a1[2];
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
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 8LL))(v2);
  *a1 = off_4A057C8;
  nullsub_801();
  j___libc_free_0((unsigned __int64)a1);
}
