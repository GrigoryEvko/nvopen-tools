// Function: sub_39636C0
// Address: 0x39636c0
//
__int64 __fastcall sub_39636C0(_QWORD *a1, __int64 a2, int **a3)
{
  __int64 v5; // r12
  int *v6; // rax
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  _QWORD *v11; // rcx
  char v12; // di

  v5 = sub_22077B0(0x30u);
  v6 = *a3;
  *(_QWORD *)(v5 + 40) = 0;
  v7 = *v6;
  *(_DWORD *)(v5 + 32) = *v6;
  v8 = sub_39635C0(a1, a2, (int *)(v5 + 32));
  v10 = v8;
  if ( v9 )
  {
    v11 = a1 + 1;
    v12 = 1;
    if ( !v8 && (_QWORD *)v9 != v11 )
      v12 = v7 < *(_DWORD *)(v9 + 32);
    sub_220F040(v12, v5, (_QWORD *)v9, v11);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5);
    return v10;
  }
}
