// Function: sub_2220380
// Address: 0x2220380
//
__int64 *__fastcall sub_2220380(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  const wchar_t *v4; // rsi
  _QWORD vars0[4]; // [rsp+0h] [rbp+0h] BYREF
  void (__fastcall *vars20)(_QWORD *); // [rsp+20h] [rbp+20h]

  v3 = *(_QWORD *)(a2 + 24);
  vars20 = 0;
  sub_2213010(v3, (__int64)vars0);
  if ( !vars20 )
    sub_426248((__int64)"uninitialized __any_string");
  v4 = (const wchar_t *)vars0[0];
  *a1 = (__int64)(a1 + 2);
  sub_221FEA0(a1, v4, (__int64)&v4[vars0[1]]);
  if ( vars20 )
    vars20(vars0);
  return a1;
}
