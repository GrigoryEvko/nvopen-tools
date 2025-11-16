// Function: sub_1F444D0
// Address: 0x1f444d0
//
_QWORD *__fastcall sub_1F444D0(__int64 a1, __int64 *a2)
{
  _QWORD **v3; // r13
  __int64 *v4; // rax
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v7; // r13
  _QWORD v8[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v9[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 520LL) != 10 )
    return sub_1F44020(a1, (__int64)a2, 1);
  v3 = *(_QWORD ***)(*(_QWORD *)(a2[1] + 56) + 40LL);
  v4 = (__int64 *)sub_16471D0(*v3, 0);
  v5 = (__int64 *)sub_1647190(v4, 0);
  v8[0] = v9;
  v8[1] = 0;
  v6 = sub_1644EA0(v5, v9, 0, 0);
  v7 = sub_1632080((__int64)v3, (__int64)"__safestack_pointer_address", 27, v6, 0);
  LOWORD(v9[0]) = 257;
  return (_QWORD *)sub_1285290(a2, *(_QWORD *)(*(_QWORD *)v7 + 24LL), v7, 0, 0, (__int64)v8, 0);
}
