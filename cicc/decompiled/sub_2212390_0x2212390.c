// Function: sub_2212390
// Address: 0x2212390
//
void __fastcall sub_2212390(_QWORD *a1)
{
  volatile signed __int32 *v2; // rdi
  signed __int32 v3; // eax

  *a1 = off_4A05138;
  v2 = (volatile signed __int32 *)a1[3];
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
  *a1 = off_4A06B40;
  sub_2254270(a1 + 2);
  nullsub_801();
  j___libc_free_0((unsigned __int64)a1);
}
