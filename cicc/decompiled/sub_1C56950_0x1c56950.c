// Function: sub_1C56950
// Address: 0x1c56950
//
__int64 __fastcall sub_1C56950(_QWORD *a1, _QWORD *a2, unsigned __int64 **a3)
{
  __int64 v5; // r12
  unsigned __int64 v6; // r14
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  _BOOL8 v11; // rdi

  v5 = sub_22077B0(72);
  v6 = **a3;
  *(_QWORD *)(v5 + 40) = 0;
  *(_QWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 32) = v6;
  *(_QWORD *)(v5 + 56) = 0;
  *(_DWORD *)(v5 + 64) = 0;
  v7 = sub_1C56850(a1, a2, (unsigned __int64 *)(v5 + 32));
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
    j___libc_free_0(0);
    j_j___libc_free_0(v5, 72);
    return (__int64)v9;
  }
}
