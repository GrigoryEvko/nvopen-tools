// Function: sub_18B4CA0
// Address: 0x18b4ca0
//
__int64 __fastcall sub_18B4CA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v3; // rax
  __int64 v4; // rdi
  __int64 *v6; // [rsp+0h] [rbp-30h] BYREF
  _BYTE v7[40]; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_15A4510(**(__int64 *****)a2, *(__int64 ***)(a1 + 48), 0);
  v3 = (__int64 *)sub_159C470(*(_QWORD *)(a1 + 64), *(_QWORD *)(a2 + 8), 0);
  v4 = *(_QWORD *)(a1 + 40);
  v7[4] = 0;
  v6 = v3;
  return sub_15A2E80(v4, v2, &v6, 1u, 0, (__int64)v7, 0);
}
