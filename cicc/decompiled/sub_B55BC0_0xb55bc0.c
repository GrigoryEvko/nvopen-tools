// Function: sub_B55BC0
// Address: 0xb55bc0
//
__int64 __fastcall sub_B55BC0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r14
  __int64 v3; // rbx
  void *v4; // r15
  __int64 v5; // rax
  __int64 v6; // r12
  char v8[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v9; // [rsp+20h] [rbp-40h]

  v1 = *(_QWORD *)(a1 - 64);
  v2 = *(_QWORD *)(a1 - 32);
  v9 = 257;
  v3 = *(unsigned int *)(a1 + 80);
  v4 = *(void **)(a1 + 72);
  v5 = sub_BD2C40(112, unk_3F1FE60);
  v6 = v5;
  if ( v5 )
    sub_B4E9E0(v5, v1, v2, v4, v3, (__int64)v8, 0, 0);
  return v6;
}
