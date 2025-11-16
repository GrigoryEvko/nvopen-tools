// Function: sub_1C9AC70
// Address: 0x1c9ac70
//
unsigned __int64 *__fastcall sub_1C9AC70(_QWORD *a1, _QWORD *a2, unsigned __int64 **a3)
{
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // r14
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  _BOOL8 v11; // rdi

  v5 = (unsigned __int64 *)sub_22077B0(56);
  v6 = **a3;
  v5[5] = 0;
  v5[6] = 0;
  v5[4] = v6;
  v7 = sub_1C9AA60(a1, a2, v5 + 4);
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && v8 != v10 )
      v11 = v6 < v8[4];
    sub_220F040(v11, v5, v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5, 56);
    return v9;
  }
}
