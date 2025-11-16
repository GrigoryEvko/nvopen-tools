// Function: sub_2E788D0
// Address: 0x2e788d0
//
__int64 __fastcall sub_2E788D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  char v3; // bl
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD v7[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  v3 = sub_AE5020(a2, v2);
  v4 = sub_9208B0(a2, v2);
  v7[1] = v5;
  v7[0] = ((1LL << v3) + ((unsigned __int64)(v4 + 7) >> 3) - 1) >> v3 << v3;
  return sub_CA1930(v7);
}
