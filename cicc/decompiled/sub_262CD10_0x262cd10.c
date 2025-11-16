// Function: sub_262CD10
// Address: 0x262cd10
//
unsigned __int64 *__fastcall sub_262CD10(_QWORD *a1, __int64 a2, unsigned __int64 **a3, _BYTE **a4)
{
  unsigned __int64 *v7; // r12
  unsigned __int64 v8; // r14
  bool v9; // zf
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // r13
  _QWORD *v13; // rcx
  char v14; // di

  v7 = (unsigned __int64 *)sub_22077B0(0x50u);
  v8 = **a3;
  v9 = **a4 == 0;
  v7[4] = v8;
  if ( v9 )
  {
    v7[6] = 0;
    v7[5] = (unsigned __int64)byte_3F871B3;
  }
  else
  {
    v7[5] = 0;
  }
  v7[7] = 0;
  v7[8] = 0;
  v7[9] = 0;
  v10 = sub_262CC10(a1, a2, v7 + 4);
  v12 = v10;
  if ( v11 )
  {
    v13 = a1 + 1;
    v14 = 1;
    if ( !v10 && v11 != v13 )
      v14 = v8 < v11[4];
    sub_220F040(v14, (__int64)v7, v11, v13);
    ++a1[5];
    return v7;
  }
  else
  {
    j_j___libc_free_0((unsigned __int64)v7);
    return v12;
  }
}
