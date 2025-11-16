// Function: sub_2214360
// Address: 0x2214360
//
__int64 __fastcall sub_2214360(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __int64 a7,
        _DWORD *a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v10; // rax
  __int64 v12; // r15
  void (*v13)(void); // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edx
  _QWORD v17[7]; // [rsp+28h] [rbp-38h] BYREF

  v10 = *a1;
  if ( a9 )
    return (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64, _QWORD))(v10 + 16))(
             a1,
             a2,
             a3,
             a4,
             a5,
             a6);
  v17[0] = &unk_4FD67D8;
  v12 = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64, _QWORD, __int64, _DWORD *, _QWORD *))(v10 + 24))(
          a1,
          a2,
          a3,
          a4,
          a5,
          a6,
          a7,
          a8,
          v17);
  if ( *a8 )
  {
    v14 = v17[0];
  }
  else
  {
    v13 = *(void (**)(void))(a10 + 32);
    if ( v13 )
      v13();
    sub_2215E70(a10, v17);
    v14 = v17[0];
    v15 = *(_QWORD *)(v17[0] - 24LL);
    *(_QWORD *)(a10 + 32) = sub_2211A80;
    *(_QWORD *)(a10 + 8) = v15;
  }
  if ( (_UNKNOWN *)(v14 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v16 = _InterlockedExchangeAdd((volatile signed __int32 *)(v14 - 8), 0xFFFFFFFF);
    }
    else
    {
      v16 = *(_DWORD *)(v14 - 8);
      *(_DWORD *)(v14 - 8) = v16 - 1;
    }
    if ( v16 <= 0 )
      j_j___libc_free_0_1(v14 - 24);
  }
  return v12;
}
