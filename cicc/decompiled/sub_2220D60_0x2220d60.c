// Function: sub_2220D60
// Address: 0x2220d60
//
void __fastcall sub_2220D60(_QWORD *a1)
{
  __int64 v2; // rax
  volatile signed __int32 *v3; // rdi
  signed __int32 v4; // eax

  *a1 = off_4A05D50;
  v2 = a1[4];
  v3 = (volatile signed __int32 *)a1[3];
  *(_QWORD *)(v2 + 24) = 0;
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
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v3 + 8LL))(v3);
  sub_220E5D0(a1);
  j___libc_free_0((unsigned __int64)a1);
}
