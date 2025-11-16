// Function: sub_90A380
// Address: 0x90a380
//
__int64 __fastcall sub_90A380(__int64 **a1)
{
  __int64 v1; // rax
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // [rsp+8h] [rbp-48h] BYREF
  char *v7; // [rsp+10h] [rbp-40h]
  __int64 v8; // [rsp+18h] [rbp-38h]
  char v9[48]; // [rsp+20h] [rbp-30h] BYREF

  v1 = sub_BA8E40(*a1, "llvm.ident", 10);
  v7 = v9;
  v2 = v1;
  v8 = 10;
  strcpy(v9, "nvcc.ident");
  v3 = **a1;
  v6 = sub_B9B140(v3, v9, 10);
  v4 = sub_B9C770(v3, &v6, 1, 0, 1);
  result = sub_B979A0(v2, v4);
  if ( v7 != v9 )
    return j_j___libc_free_0(v7, *(_QWORD *)v9 + 1LL);
  return result;
}
