// Function: sub_2213F20
// Address: 0x2213f20
//
void __fastcall sub_2213F20(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7)
{
  void (__fastcall *v9)(__int64); // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // rdi
  int v13; // edx
  int v14; // edx
  char v16; // [rsp+1Eh] [rbp-4Ah] BYREF
  __int64 v17; // [rsp+20h] [rbp-48h] BYREF
  _QWORD v18[8]; // [rsp+28h] [rbp-40h] BYREF

  sub_2215F80(&v17, a6, a7, &v16);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD, _QWORD, __int64 *))(*(_QWORD *)a1 + 24LL))(
    v18,
    a1,
    a3,
    a4,
    a5,
    &v17);
  v9 = *(void (__fastcall **)(__int64))(a2 + 32);
  if ( v9 )
    v9(a2);
  sub_2215E70(a2, v18);
  v10 = v18[0];
  v11 = *(_QWORD *)(v18[0] - 24LL);
  *(_QWORD *)(a2 + 32) = sub_2211A80;
  *(_QWORD *)(a2 + 8) = v11;
  if ( (_UNKNOWN *)(v10 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v13 = _InterlockedExchangeAdd((volatile signed __int32 *)(v10 - 8), 0xFFFFFFFF);
    }
    else
    {
      v13 = *(_DWORD *)(v10 - 8);
      *(_DWORD *)(v10 - 8) = v13 - 1;
    }
    if ( v13 <= 0 )
      j_j___libc_free_0_1(v10 - 24);
  }
  v12 = v17 - 24;
  if ( (_UNKNOWN *)(v17 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v14 = _InterlockedExchangeAdd((volatile signed __int32 *)(v17 - 8), 0xFFFFFFFF);
    }
    else
    {
      v14 = *(_DWORD *)(v17 - 8);
      *(_DWORD *)(v17 - 8) = v14 - 1;
    }
    if ( v14 <= 0 )
      j_j___libc_free_0_1(v12);
  }
}
