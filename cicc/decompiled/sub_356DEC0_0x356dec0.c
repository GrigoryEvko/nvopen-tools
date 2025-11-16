// Function: sub_356DEC0
// Address: 0x356dec0
//
__int64 __fastcall sub_356DEC0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // rax
  _QWORD *v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9

  a1[1] = a3;
  a1[2] = a4;
  a1[3] = a5;
  v5 = *(_QWORD *)(a2 + 328);
  v6 = sub_22077B0(0x70u);
  v7 = (_QWORD *)v6;
  if ( v6 )
    sub_3568730(v6, v5, 0, (__int64)a1, a1[1], 0);
  a1[4] = (__int64)v7;
  sub_35680F0((__int64)a1, v7);
  return sub_356DE20((__int64)a1, a2, v8, v9, v10, v11);
}
