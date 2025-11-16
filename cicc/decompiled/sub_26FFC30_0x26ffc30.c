// Function: sub_26FFC30
// Address: 0x26ffc30
//
unsigned __int64 *__fastcall sub_26FFC30(_QWORD *a1, _QWORD *a2, unsigned __int64 **a3)
{
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // r15
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r14
  _QWORD *v10; // rcx
  char v11; // di
  unsigned __int64 v13; // rdi

  v5 = (unsigned __int64 *)sub_22077B0(0x80u);
  v6 = **a3;
  v5[4] = v6;
  memset(v5 + 5, 0, 0x58u);
  v5[6] = (unsigned __int64)(v5 + 8);
  v5[13] = (unsigned __int64)(v5 + 11);
  v5[14] = (unsigned __int64)(v5 + 11);
  v7 = sub_9D7FB0(a1, a2, v5 + 4);
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && v8 != v10 )
      v11 = v6 < v8[4];
    sub_220F040(v11, (__int64)v5, v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    sub_26F81E0(0);
    v13 = v5[6];
    if ( v5 + 8 != (unsigned __int64 *)v13 )
      j_j___libc_free_0(v13);
    j_j___libc_free_0((unsigned __int64)v5);
    return v9;
  }
}
