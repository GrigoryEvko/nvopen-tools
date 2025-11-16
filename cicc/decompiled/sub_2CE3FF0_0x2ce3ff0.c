// Function: sub_2CE3FF0
// Address: 0x2ce3ff0
//
__int64 __fastcall sub_2CE3FF0(_QWORD *a1, __int64 a2, unsigned __int64 **a3)
{
  __int64 v5; // r12
  unsigned __int64 *v6; // rax
  unsigned __int64 v7; // r14
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  _QWORD *v10; // r13
  _QWORD *v11; // rcx
  char v12; // di

  v5 = sub_22077B0(0x30u);
  v6 = *a3;
  *(_DWORD *)(v5 + 40) = 0;
  v7 = *v6;
  *(_QWORD *)(v5 + 32) = *v6;
  v8 = sub_2CE3EF0(a1, a2, (unsigned __int64 *)(v5 + 32));
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
    j_j___libc_free_0(v5);
    return (__int64)v10;
  }
}
