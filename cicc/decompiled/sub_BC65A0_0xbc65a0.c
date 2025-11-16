// Function: sub_BC65A0
// Address: 0xbc65a0
//
__int64 __fastcall sub_BC65A0(_QWORD *a1, __int64 a2, int **a3)
{
  __int64 v5; // r12
  int *v6; // rax
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  _QWORD *v11; // rcx
  _BOOL8 v12; // rdi

  v5 = sub_22077B0(40);
  v6 = *a3;
  *(_DWORD *)(v5 + 36) = 0;
  v7 = *v6;
  *(_DWORD *)(v5 + 32) = *v6;
  v8 = sub_9814F0(a1, a2, (int *)(v5 + 32));
  v10 = v8;
  if ( v9 )
  {
    v11 = a1 + 1;
    v12 = 1;
    if ( !v8 && (_QWORD *)v9 != v11 )
      v12 = v7 < *(_DWORD *)(v9 + 32);
    sub_220F040(v12, v5, v9, v11);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5, 40);
    return v10;
  }
}
