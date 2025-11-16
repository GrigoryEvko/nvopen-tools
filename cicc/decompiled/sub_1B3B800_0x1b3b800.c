// Function: sub_1B3B800
// Address: 0x1b3b800
//
bool __fastcall sub_1B3B800(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rdi
  __int64 v4; // rax
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v3 = *(_QWORD **)a3;
  v4 = *(unsigned int *)(a3 + 8);
  v6 = a2;
  return &v3[v4] != sub_1B3B740(v3, (__int64)&v3[v4], &v6);
}
