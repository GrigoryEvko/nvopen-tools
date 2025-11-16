// Function: sub_3896710
// Address: 0x3896710
//
__int64 __fastcall sub_3896710(_QWORD *a1, __int64 a2, unsigned int **a3)
{
  __int64 v5; // r12
  unsigned int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // rcx
  char v11; // di

  v5 = sub_22077B0(0x88u);
  v6 = **a3;
  *(_DWORD *)(v5 + 32) = v6;
  memset((void *)(v5 + 40), 0, 0x60u);
  *(_QWORD *)(v5 + 72) = v5 + 56;
  *(_QWORD *)(v5 + 80) = v5 + 56;
  v7 = sub_3896610(a1, a2, (unsigned int *)(v5 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && (_QWORD *)v8 != v10 )
      v11 = v6 < *(_DWORD *)(v8 + 32);
    sub_220F040(v11, v5, (_QWORD *)v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    sub_3887AD0(0);
    j_j___libc_free_0(v5);
    return v9;
  }
}
