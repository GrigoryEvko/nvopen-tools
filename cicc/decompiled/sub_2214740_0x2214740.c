// Function: sub_2214740
// Address: 0x2214740
//
__int64 __fastcall sub_2214740(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 a5,
        unsigned int a6,
        __int128 a7,
        _QWORD *a8)
{
  __int64 result; // rax
  unsigned __int64 v11; // rdi
  int v12; // esi
  __int64 v15; // [rsp+10h] [rbp-50h]
  char v16; // [rsp+27h] [rbp-39h] BYREF
  _QWORD v17[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( !a8 )
    return (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 16LL))(
             a1,
             a2,
             a3,
             a4,
             a5);
  if ( !a8[4] )
    sub_426248((__int64)"uninitialized __any_string");
  sub_2216C30(v17, *a8, a8[1], &v16);
  result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64, _QWORD, _QWORD *))(*(_QWORD *)a1 + 24LL))(
             a1,
             a2,
             a3,
             a4,
             a5,
             a6,
             v17);
  v11 = v17[0] - 24LL;
  if ( (_UNKNOWN *)(v17[0] - 24LL) != &unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v12 = _InterlockedExchangeAdd((volatile signed __int32 *)(v17[0] - 8LL), 0xFFFFFFFF);
    }
    else
    {
      v12 = *(_DWORD *)(v17[0] - 8LL);
      *(_DWORD *)(v17[0] - 8LL) = v12 - 1;
    }
    if ( v12 <= 0 )
    {
      v15 = result;
      j_j___libc_free_0_2(v11);
      return v15;
    }
  }
  return result;
}
