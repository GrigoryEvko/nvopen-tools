// Function: sub_2222970
// Address: 0x2222970
//
__int64 __fastcall sub_2222970(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 a5,
        unsigned int a6,
        __int128 a7,
        __int64 a8)
{
  const wchar_t *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v17[2]; // [rsp+20h] [rbp-50h] BYREF
  _BYTE v18[64]; // [rsp+30h] [rbp-40h] BYREF

  if ( !a8 )
    return (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a1 + 16LL))(a1, a2, a3, a4);
  if ( !*(_QWORD *)(a8 + 32) )
    sub_426248((__int64)"uninitialized __any_string");
  v11 = *(const wchar_t **)a8;
  v12 = *(_QWORD *)(a8 + 8);
  v17[0] = (__int64)v18;
  sub_221FEA0(v17, v11, (__int64)&v11[v12]);
  v13 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 24LL))(
          a1,
          a2,
          a3,
          a4,
          a5,
          a6,
          v17);
  if ( (_BYTE *)v17[0] != v18 )
    j___libc_free_0(v17[0]);
  return v13;
}
