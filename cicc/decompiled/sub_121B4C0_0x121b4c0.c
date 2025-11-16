// Function: sub_121B4C0
// Address: 0x121b4c0
//
__int64 __fastcall sub_121B4C0(_QWORD *a1, __int64 a2, unsigned int **a3)
{
  __int64 v5; // r12
  unsigned int *v6; // rax
  unsigned int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  _QWORD *v11; // rcx
  _BOOL8 v12; // rdi

  v5 = sub_22077B0(48);
  v6 = *a3;
  *(_QWORD *)(v5 + 40) = 0;
  v7 = *v6;
  *(_DWORD *)(v5 + 32) = *v6;
  v8 = sub_121B3C0(a1, a2, (unsigned int *)(v5 + 32));
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
    j_j___libc_free_0(v5, 48);
    return v10;
  }
}
