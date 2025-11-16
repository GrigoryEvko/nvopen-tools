// Function: sub_1E064E0
// Address: 0x1e064e0
//
__int64 __fastcall sub_1E064E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v6; // r12
  __int64 *v7; // rax
  __int64 v8; // r13
  __int64 *v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // [rsp+8h] [rbp-48h] BYREF
  __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  char *v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v13 = a2;
  v4 = sub_1E05220(a3, a2);
  if ( !v4 )
  {
    v14 = a2;
    if ( (unsigned __int8)sub_1E060E0(a1 + 24, &v14, v15)
      && v15[0] != (char *)(*(_QWORD *)(a1 + 32) + 72LL * *(unsigned int *)(a1 + 48)) )
    {
      v4 = *((_QWORD *)v15[0] + 4);
    }
    v6 = sub_1E064E0(a1, v4, a3);
    sub_1E04AB0(&v14, v13, v6);
    v15[0] = (char *)v14;
    sub_1E06030(v6 + 24, v15);
    v4 = v14;
    v14 = 0;
    v7 = sub_1E063B0(a3 + 24, &v13);
    v8 = v7[1];
    v9 = v7;
    v7[1] = v4;
    if ( v8 )
    {
      v10 = *(_QWORD *)(v8 + 24);
      if ( v10 )
        j_j___libc_free_0(v10, *(_QWORD *)(v8 + 40) - v10);
      j_j___libc_free_0(v8, 56);
      v4 = v9[1];
    }
    v11 = v14;
    if ( v14 )
    {
      v12 = *(_QWORD *)(v14 + 24);
      if ( v12 )
        j_j___libc_free_0(v12, *(_QWORD *)(v14 + 40) - v12);
      j_j___libc_free_0(v11, 56);
    }
  }
  return v4;
}
