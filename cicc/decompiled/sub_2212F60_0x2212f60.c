// Function: sub_2212F60
// Address: 0x2212f60
//
void __fastcall sub_2212F60(__int64 a1, __int64 a2)
{
  void (__fastcall *v2)(__int64); // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rdi
  bool v6; // zf
  int v7; // edx
  _QWORD v8[4]; // [rsp+8h] [rbp-20h] BYREF

  (*(void (__fastcall **)(_QWORD *, __int64))(*(_QWORD *)a1 + 24LL))(v8, a1);
  v2 = *(void (__fastcall **)(__int64))(a2 + 32);
  if ( v2 )
    v2(a2);
  sub_2215E70(a2, v8);
  v3 = v8[0];
  v4 = *(_QWORD *)(v8[0] - 24LL);
  v5 = v8[0] - 24LL;
  v6 = v8[0] - 24LL == (_QWORD)&unk_4FD67C0;
  *(_QWORD *)(a2 + 32) = sub_2211A80;
  *(_QWORD *)(a2 + 8) = v4;
  if ( !v6 )
  {
    if ( &_pthread_key_create )
    {
      v7 = _InterlockedExchangeAdd((volatile signed __int32 *)(v3 - 8), 0xFFFFFFFF);
    }
    else
    {
      v7 = *(_DWORD *)(v3 - 8);
      *(_DWORD *)(v3 - 8) = v7 - 1;
    }
    if ( v7 <= 0 )
      j_j___libc_free_0_1(v5);
  }
}
