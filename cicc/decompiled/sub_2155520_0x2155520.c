// Function: sub_2155520
// Address: 0x2155520
//
__int64 __fastcall sub_2155520(_QWORD *a1, __int64 a2, unsigned int **a3)
{
  __int64 v5; // r12
  unsigned int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // rcx
  _BOOL8 v11; // rdi

  v5 = sub_22077B0(64);
  v6 = **a3;
  *(_BYTE *)(v5 + 56) = 0;
  *(_QWORD *)(v5 + 40) = 0;
  *(_DWORD *)(v5 + 32) = v6;
  *(_QWORD *)(v5 + 48) = 0;
  v7 = sub_2155420(a1, a2, (unsigned int *)(v5 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && (_QWORD *)v8 != v10 )
      v11 = v6 > *(_DWORD *)(v8 + 32);
    sub_220F040(v11, v5, v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5, 64);
    return v9;
  }
}
