// Function: sub_1BF1CC0
// Address: 0x1bf1cc0
//
__int64 __fastcall sub_1BF1CC0(__int64 a1, void *a2, size_t a3, unsigned int a4)
{
  __int64 *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v6 = (__int64 *)sub_157E9C0(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 32LL));
  v10[0] = sub_161FF10(v6, a2, a3);
  v7 = sub_1643350(v6);
  v8 = sub_159C470(v7, a4, 0);
  v10[1] = (__int64)sub_1624210(v8);
  return sub_1627350(v6, v10, (__int64 *)2, 0, 1);
}
