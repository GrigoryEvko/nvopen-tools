// Function: sub_1E691E0
// Address: 0x1e691e0
//
__int64 __fastcall sub_1E691E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // rax
  _QWORD *v7; // r13

  a1[1] = a3;
  a1[2] = a4;
  a1[3] = a5;
  v5 = *(_QWORD *)(a2 + 328);
  v6 = sub_22077B0(112);
  v7 = (_QWORD *)v6;
  if ( v6 )
    sub_1E63500(v6, v5, 0, (__int64)a1, a1[1], 0);
  a1[4] = (__int64)v7;
  sub_1E62EE0((__int64)a1, v7);
  return sub_1E690F0((__int64)a1, a2);
}
