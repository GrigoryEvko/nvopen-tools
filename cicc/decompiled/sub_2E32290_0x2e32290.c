// Function: sub_2E32290
// Address: 0x2e32290
//
bool __fastcall sub_2E32290(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r8
  __int64 v5; // [rsp+8h] [rbp-8h] BYREF

  v2 = *(unsigned int *)(a1 + 72);
  v3 = *(_QWORD **)(a1 + 64);
  v5 = a2;
  return &v3[v2] != sub_2E2FCE0(v3, (__int64)&v3[v2], &v5);
}
