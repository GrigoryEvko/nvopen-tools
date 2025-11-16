// Function: sub_15FB4C0
// Address: 0x15fb4c0
//
__int64 __fastcall sub_15FB4C0(int a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rcx
  __int64 v9; // rax

  v6 = a5 + 40;
  v7 = sub_15FB440(a1, a2, a3, a4, 0);
  sub_157E9D0(a5 + 40, v7);
  v8 = *(_QWORD *)(a5 + 40);
  v9 = *(_QWORD *)(v7 + 24);
  *(_QWORD *)(v7 + 32) = v6;
  v8 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v7 + 24) = v8 | v9 & 7;
  *(_QWORD *)(v8 + 8) = v7 + 24;
  *(_QWORD *)(a5 + 40) = *(_QWORD *)(a5 + 40) & 7LL | (v7 + 24);
  return v7;
}
