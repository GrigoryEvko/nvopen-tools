// Function: sub_2292130
// Address: 0x2292130
//
bool __fastcall sub_2292130(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, _BYTE *a5)
{
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // r8
  _QWORD *v9; // r14
  bool v10; // r8
  bool result; // al
  __int64 *v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 *v16; // rax
  __int64 v17; // r8
  _QWORD *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  _QWORD *v21; // rax
  __int64 *v22; // [rsp+0h] [rbp-70h]
  __int64 *v23; // [rsp+8h] [rbp-68h]
  unsigned __int64 v26[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v27[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = sub_228CE20(a4);
  v9 = sub_2291EA0(a1, *a2, v6, v7, v8);
  v10 = sub_D968A0((__int64)v9);
  result = 0;
  if ( !v10 )
  {
    v22 = *(__int64 **)(a1 + 8);
    v27[1] = sub_228CE10(a4);
    v26[0] = (unsigned __int64)v27;
    v27[0] = v9;
    v26[1] = 0x200000002LL;
    v12 = sub_DC8BD0(v22, (__int64)v26, 0, 0);
    v13 = (__int64)v12;
    if ( (_QWORD *)v26[0] != v27 )
    {
      v23 = v12;
      _libc_free(v26[0]);
      v13 = (__int64)v23;
    }
    v14 = sub_DCC810(*(__int64 **)(a1 + 8), *a2, v13, 0, 0);
    *a2 = (__int64)v14;
    *a2 = (__int64)sub_2291F00(a1, (__int64)v14, v6, v15);
    v16 = sub_DCAF50(*(__int64 **)(a1 + 8), (__int64)v9, 0);
    v18 = sub_2291FC0(a1, *a3, v6, (__int64)v16, v17);
    *a3 = (__int64)v18;
    v21 = sub_2291EA0(a1, (__int64)v18, v6, v19, v20);
    result = sub_D968A0((__int64)v21);
    if ( !result )
    {
      *a5 = 0;
      return 1;
    }
  }
  return result;
}
