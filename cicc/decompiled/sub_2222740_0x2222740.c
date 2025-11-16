// Function: sub_2222740
// Address: 0x2222740
//
__int64 __fastcall sub_2222740(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __int64 a7,
        _DWORD *a8,
        __int64 a9,
        __int64 *a10)
{
  __int64 v10; // rax
  __int64 v12; // r14
  void (__fastcall *v13)(__int64 *); // rax
  const wchar_t *v14; // rsi
  const wchar_t *v15; // [rsp+10h] [rbp-50h] BYREF
  __int64 v16; // [rsp+18h] [rbp-48h]
  _DWORD v17[16]; // [rsp+20h] [rbp-40h] BYREF

  v10 = *a1;
  if ( a9 )
    return (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64, _QWORD))(v10 + 16))(
             a1,
             a2,
             a3,
             a4,
             a5,
             a6);
  v16 = 0;
  v15 = v17;
  v17[0] = 0;
  v12 = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64, _QWORD, __int64, _DWORD *, const wchar_t **))(v10 + 24))(
          a1,
          a2,
          a3,
          a4,
          a5,
          a6,
          a7,
          a8,
          &v15);
  if ( !*a8 )
  {
    v13 = (void (__fastcall *)(__int64 *))a10[4];
    if ( v13 )
      v13(a10);
    v14 = v15;
    *a10 = (__int64)(a10 + 2);
    sub_2220100(a10, v14, (__int64)&v14[v16]);
    a10[4] = (__int64)sub_221F8F0;
  }
  if ( v15 != v17 )
    j___libc_free_0((unsigned __int64)v15);
  return v12;
}
