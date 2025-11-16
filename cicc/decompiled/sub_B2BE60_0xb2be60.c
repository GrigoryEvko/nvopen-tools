// Function: sub_B2BE60
// Address: 0xb2be60
//
__int64 __fastcall sub_B2BE60(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  int v4; // r12d
  __int64 *v5; // rax
  unsigned __int64 v6; // r8
  __int64 result; // rax
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD *)(a1 + 24);
  v4 = *(_DWORD *)(a1 + 32);
  v8[0] = *(_QWORD *)(v3 + 120);
  v5 = (__int64 *)sub_B2BE50(v3);
  v6 = sub_A7A440(v8, v5, v4 + 1, a2);
  result = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(result + 120) = v6;
  return result;
}
