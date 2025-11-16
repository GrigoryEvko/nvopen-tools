// Function: sub_2220400
// Address: 0x2220400
//
__int64 *__fastcall sub_2220400(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        __int64 *a6)
{
  __int64 v7; // rdi
  __int64 v8; // r9
  const wchar_t *v9; // rsi
  __int64 v11; // [rsp-8h] [rbp-50h]
  _QWORD v12[4]; // [rsp+8h] [rbp-40h] BYREF
  void (__fastcall *v13)(_QWORD *); // [rsp+28h] [rbp-20h]

  v7 = *(_QWORD *)(a2 + 32);
  v11 = a6[1];
  v8 = *a6;
  v13 = 0;
  sub_2214070(v7, (__int64)v12, a3, a4, a5, v8, v11);
  if ( !v13 )
    sub_426248((__int64)"uninitialized __any_string");
  v9 = (const wchar_t *)v12[0];
  *a1 = (__int64)(a1 + 2);
  sub_221FEA0(a1, v9, (__int64)&v9[v12[1]]);
  if ( v13 )
    v13(v12);
  return a1;
}
