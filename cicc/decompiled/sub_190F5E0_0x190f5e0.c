// Function: sub_190F5E0
// Address: 0x190f5e0
//
__int64 __fastcall sub_190F5E0(_QWORD *a1, _QWORD *a2, unsigned __int64 **a3)
{
  __int64 v5; // r12
  unsigned __int64 *v6; // rax
  unsigned __int64 v7; // r14
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  _QWORD *v10; // r13
  _QWORD *v11; // rcx
  _BOOL8 v12; // rdi

  v5 = sub_22077B0(48);
  v6 = *a3;
  *(_QWORD *)(v5 + 40) = 0;
  v7 = *v6;
  *(_QWORD *)(v5 + 32) = *v6;
  v8 = sub_190F4E0(a1, a2, (unsigned __int64 *)(v5 + 32));
  v10 = v8;
  if ( v9 )
  {
    v11 = a1 + 1;
    v12 = 1;
    if ( !v8 && v9 != v11 )
      v12 = v7 < v9[4];
    sub_220F040(v12, v5, v9, v11);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5, 48);
    return (__int64)v10;
  }
}
