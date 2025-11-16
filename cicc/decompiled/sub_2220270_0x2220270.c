// Function: sub_2220270
// Address: 0x2220270
//
__int64 *__fastcall sub_2220270(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rsi
  __int64 v5; // rdx
  _QWORD vars0[4]; // [rsp+0h] [rbp+0h] BYREF
  void (__fastcall *vars20)(_QWORD *); // [rsp+20h] [rbp+20h]

  v3 = *(_QWORD *)(a2 + 24);
  vars20 = 0;
  sub_2212F60(v3, (__int64)vars0);
  if ( !vars20 )
    sub_426248((__int64)"uninitialized __any_string");
  v4 = (_BYTE *)vars0[0];
  v5 = vars0[1];
  *a1 = (__int64)(a1 + 2);
  sub_221FC40(a1, v4, (__int64)&v4[v5]);
  if ( vars20 )
    vars20(vars0);
  return a1;
}
