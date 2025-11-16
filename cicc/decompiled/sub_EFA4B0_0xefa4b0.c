// Function: sub_EFA4B0
// Address: 0xefa4b0
//
__int64 __fastcall sub_EFA4B0(_QWORD *a1, __int64 a2, __int64 **a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // rcx
  _BOOL8 v11; // rdi
  unsigned int v13; // eax

  v5 = sub_22077B0(104);
  v6 = **a3;
  *(_OWORD *)(v5 + 40) = 0;
  *(_OWORD *)(v5 + 72) = 0;
  *(_QWORD *)(v5 + 32) = v6;
  *(_QWORD *)(v5 + 48) = v5 + 96;
  *(_QWORD *)(v5 + 56) = 1;
  *(_QWORD *)(v5 + 64) = 0;
  *(_DWORD *)(v5 + 80) = 1065353216;
  *(_OWORD *)(v5 + 88) = 0;
  v7 = sub_C1D150(a1, a2, (unsigned int *)(v5 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && (_QWORD *)v8 != v10 )
    {
      v13 = *(_DWORD *)(v8 + 32);
      if ( *(_DWORD *)(v5 + 32) >= v13 )
      {
        v11 = 0;
        if ( *(_DWORD *)(v5 + 32) == v13 )
          v11 = *(_DWORD *)(v5 + 36) < *(_DWORD *)(v8 + 36);
      }
    }
    sub_220F040(v11, v5, v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5, 104);
    return v9;
  }
}
