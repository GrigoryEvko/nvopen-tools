// Function: sub_2E33650
// Address: 0x2e33650
//
__int64 *__fastcall sub_2E33650(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // rax
  __int64 *v5; // rax
  __int64 v6; // r8
  char v7; // r9
  __int64 v9; // [rsp+8h] [rbp-8h] BYREF

  v3 = *(_QWORD **)(a1 + 112);
  v4 = *(unsigned int *)(a1 + 120);
  v9 = a2;
  v5 = sub_2E2FDA0(v3, (__int64)&v3[v4], &v9);
  return sub_2E33590(v6, v5, v7);
}
