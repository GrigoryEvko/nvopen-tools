// Function: sub_22202F0
// Address: 0x22202f0
//
__int64 *__fastcall sub_22202F0(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        __int64 *a6)
{
  __int64 v7; // rdi
  __int64 v8; // r9
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  __int64 v12; // [rsp-8h] [rbp-50h]
  _QWORD v13[4]; // [rsp+8h] [rbp-40h] BYREF
  void (__fastcall *v14)(_QWORD *); // [rsp+28h] [rbp-20h]

  v7 = *(_QWORD *)(a2 + 32);
  v12 = a6[1];
  v8 = *a6;
  v14 = 0;
  sub_2213F20(v7, (__int64)v13, a3, a4, a5, v8, v12);
  if ( !v14 )
    sub_426248((__int64)"uninitialized __any_string");
  v9 = (_BYTE *)v13[0];
  v10 = v13[1];
  *a1 = (__int64)(a1 + 2);
  sub_221FC40(a1, v9, (__int64)&v9[v10]);
  if ( v14 )
    v14(v13);
  return a1;
}
