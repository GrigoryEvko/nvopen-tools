// Function: sub_22E24D0
// Address: 0x22e24d0
//
__int64 __fastcall sub_22E24D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9

  a1[1] = a3;
  a1[2] = a4;
  a1[3] = a5;
  v5 = *(_QWORD *)(a2 + 80);
  if ( v5 )
    v5 -= 24;
  v6 = sub_22077B0(0x70u);
  v7 = (_QWORD *)v6;
  if ( v6 )
    sub_22DBFA0(v6, v5, 0, (__int64)a1, a1[1], 0);
  a1[4] = (__int64)v7;
  sub_22DB840((__int64)a1, v7);
  return sub_22E2440((__int64)a1, a2, v8, v9, v10, v11);
}
