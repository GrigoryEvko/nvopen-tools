// Function: sub_C54F20
// Address: 0xc54f20
//
__int64 __fastcall sub_C54F20(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v4 = sub_CB7210(a1, a2);
  v5 = *(_QWORD *)(a2 + 24);
  v10[2] = 2;
  v6 = v4;
  v7 = *(_QWORD *)(a2 + 32);
  v10[0] = v5;
  v10[1] = v7;
  sub_C51AE0(v6, (__int64)v10);
  v8 = sub_CB7210(v6, v10);
  return sub_CB69B0(v8, (unsigned int)(a3 - *(_DWORD *)(a2 + 32)));
}
