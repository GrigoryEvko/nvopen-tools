// Function: sub_18BE220
// Address: 0x18be220
//
unsigned __int64 *__fastcall sub_18BE220(_QWORD *a1, _QWORD *a2, unsigned __int64 **a3)
{
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // r15
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r14
  _QWORD *v10; // rcx
  _BOOL8 v11; // rdi
  unsigned __int64 *v13; // rdi

  v5 = (unsigned __int64 *)sub_22077B0(128);
  v6 = **a3;
  v5[4] = v6;
  memset(v5 + 5, 0, 0x58u);
  v5[6] = (unsigned __int64)(v5 + 8);
  v5[13] = (unsigned __int64)(v5 + 11);
  v5[14] = (unsigned __int64)(v5 + 11);
  v7 = sub_14F7B80(a1, a2, v5 + 4);
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
    sub_18B5550(0);
    v13 = (unsigned __int64 *)v5[6];
    if ( v5 + 8 != v13 )
      j_j___libc_free_0(v13, v5[8] + 1);
    j_j___libc_free_0(v5, 128);
    return v9;
  }
}
