// Function: sub_2213DC0
// Address: 0x2213dc0
//
__int64 __fastcall sub_2213DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned __int64 v6; // rdi
  int v7; // ecx
  unsigned int v8; // [rsp+Ch] [rbp-3Ch]
  char v9; // [rsp+17h] [rbp-31h] BYREF
  _QWORD v10[6]; // [rsp+18h] [rbp-30h] BYREF

  sub_2215F80(v10, a2, a3, &v9);
  result = (*(__int64 (__fastcall **)(__int64, _QWORD *, __int64))(*(_QWORD *)a1 + 16LL))(a1, v10, a4);
  v6 = v10[0] - 24LL;
  if ( (_UNKNOWN *)(v10[0] - 24LL) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v7 = _InterlockedExchangeAdd((volatile signed __int32 *)(v10[0] - 8LL), 0xFFFFFFFF);
    }
    else
    {
      v7 = *(_DWORD *)(v10[0] - 8LL);
      *(_DWORD *)(v10[0] - 8LL) = v7 - 1;
    }
    if ( v7 <= 0 )
    {
      v8 = result;
      j_j___libc_free_0_1(v6);
      return v8;
    }
  }
  return result;
}
