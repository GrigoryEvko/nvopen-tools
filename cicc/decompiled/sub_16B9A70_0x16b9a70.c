// Function: sub_16B9A70
// Address: 0x16b9a70
//
unsigned __int64 *__fastcall sub_16B9A70(_QWORD *a1, _QWORD *a2, unsigned __int64 **a3)
{
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // r14
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  _BOOL8 v11; // rdi

  v5 = (unsigned __int64 *)sub_22077B0(64);
  v6 = **a3;
  v5[5] = 0;
  v5[6] = 0;
  v5[4] = v6;
  v5[7] = 0;
  v7 = sub_16B9970(a1, a2, v5 + 4);
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
    j_j___libc_free_0(v5, 64);
    return v9;
  }
}
