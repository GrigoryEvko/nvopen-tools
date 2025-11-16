// Function: sub_3115870
// Address: 0x3115870
//
__int64 __fastcall sub_3115870(_QWORD *a1, __int64 a2, unsigned int **a3)
{
  __int64 v5; // r12
  unsigned int *v6; // rax
  unsigned int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  _QWORD *v11; // rcx
  char v12; // di

  v5 = sub_22077B0(0x50u);
  v6 = *a3;
  *(_OWORD *)(v5 + 40) = 0;
  v7 = *v6;
  *(_OWORD *)(v5 + 56) = 0;
  *(_QWORD *)(v5 + 72) = 0;
  *(_DWORD *)(v5 + 32) = v7;
  v8 = sub_3115770(a1, a2, (unsigned int *)(v5 + 32));
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
