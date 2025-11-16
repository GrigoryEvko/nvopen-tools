// Function: sub_15A4320
// Address: 0x15a4320
//
__int64 __fastcall sub_15A4320(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 *v2; // r12
  __int64 **v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  __int64 **v8; // rax
  __int64 *v10; // [rsp+0h] [rbp-20h] BYREF
  _BYTE v11[24]; // [rsp+8h] [rbp-18h] BYREF

  v1 = sub_1643350(*a1);
  v2 = (__int64 *)sub_159C470(v1, 1, 0);
  v3 = (__int64 **)sub_1646BA0(a1, 0);
  v6 = sub_15A06D0(v3, 0, v4, v5);
  v10 = v2;
  v11[4] = 0;
  v7 = sub_15A2E80((__int64)a1, v6, &v10, 1u, 0, (__int64)v11, 0);
  v8 = (__int64 **)sub_1643360(*a1);
  return sub_15A4180(v7, v8, 0);
}
