// Function: sub_15D0040
// Address: 0x15d0040
//
__int64 __fastcall sub_15D0040(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r12
  __int64 *v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 *v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h] BYREF
  __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  char *v15; // [rsp+18h] [rbp-28h] BYREF

  v13 = a2;
  v3 = sub_15CC510(a1, a3);
  *(_BYTE *)(a1 + 72) = 0;
  v4 = v3;
  sub_15CC0B0(&v14, v13, v3);
  v15 = (char *)v14;
  sub_15CE4A0(v4 + 24, &v15);
  v5 = v14;
  v14 = 0;
  v6 = sub_15CFF10(a1 + 24, &v13);
  v7 = v6[1];
  v6[1] = v5;
  if ( v7 )
  {
    v8 = *(_QWORD *)(v7 + 24);
    v9 = v6;
    if ( v8 )
      j_j___libc_free_0(v8, *(_QWORD *)(v7 + 40) - v8);
    j_j___libc_free_0(v7, 56);
    v5 = v9[1];
  }
  v10 = v14;
  if ( v14 )
  {
    v11 = *(_QWORD *)(v14 + 24);
    if ( v11 )
      j_j___libc_free_0(v11, *(_QWORD *)(v14 + 40) - v11);
    j_j___libc_free_0(v10, 56);
  }
  return v5;
}
