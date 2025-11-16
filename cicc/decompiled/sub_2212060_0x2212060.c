// Function: sub_2212060
// Address: 0x2212060
//
void __fastcall sub_2212060(_QWORD *a1)
{
  volatile signed __int32 *v2; // rdi
  signed __int32 v3; // eax

  *a1 = off_49D2868;
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
  *a1 = off_4A07CC8;
  nullsub_801();
  j___libc_free_0((unsigned __int64)a1);
}
