// Function: sub_CA37D0
// Address: 0xca37d0
//
__int64 __fastcall sub_CA37D0(__int64 a1, __int64 a2, void **a3)
{
  int v4; // r15d
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v9; // [rsp+Ch] [rbp-44h]
  __int64 v10; // [rsp+10h] [rbp-40h]
  int v11; // [rsp+18h] [rbp-38h]
  int v12; // [rsp+1Ch] [rbp-34h]

  v9 = *(_DWORD *)(a2 + 44);
  v4 = *(_DWORD *)(a2 + 24);
  v11 = *(_DWORD *)(a2 + 40);
  v10 = *(_QWORD *)(a2 + 32);
  v12 = *(_DWORD *)(a2 + 28);
  v5 = sub_C82280(a2);
  v6 = sub_C82290(a2);
  sub_CA3710(a1, a3, v6, v7, v5, v4, v12, v10, v11, v9);
  return a1;
}
