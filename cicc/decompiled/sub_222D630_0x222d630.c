// Function: sub_222D630
// Address: 0x222d630
//
void __fastcall sub_222D630(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rbp
  volatile signed __int32 *v2; // rdi
  signed __int32 v3; // eax
  void (*v4)(void); // rdx

  v1 = *a1;
  if ( *a1 )
  {
    v2 = *(volatile signed __int32 **)(v1 + 24);
    if ( v2 )
    {
      if ( &_pthread_key_create )
      {
        v3 = _InterlockedExchangeAdd(v2 + 3, 0xFFFFFFFF);
      }
      else
      {
        v3 = *((_DWORD *)v2 + 3);
        *((_DWORD *)v2 + 3) = v3 - 1;
      }
      if ( v3 == 1 )
      {
        v4 = *(void (**)(void))(*(_QWORD *)v2 + 24LL);
        if ( (char *)v4 == (char *)sub_9C26E0 )
          (*(void (**)(void))(*(_QWORD *)v2 + 8LL))();
        else
          v4();
      }
    }
    j___libc_free_0(v1);
  }
}
