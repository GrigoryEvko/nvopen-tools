// Function: sub_2C96770
// Address: 0x2c96770
//
__int64 __fastcall sub_2C96770(_QWORD *a1, __int64 a2, unsigned __int64 **a3)
{
  __int64 v5; // r12
  unsigned __int64 v6; // r14
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  char v11; // di

  v5 = sub_22077B0(0x48u);
  v6 = **a3;
  *(_QWORD *)(v5 + 40) = 0;
  *(_QWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 32) = v6;
  *(_QWORD *)(v5 + 56) = 0;
  *(_DWORD *)(v5 + 64) = 0;
  v7 = sub_2C96670(a1, a2, (unsigned __int64 *)(v5 + 32));
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
    sub_C7D6A0(0, 0, 8);
    j_j___libc_free_0(v5);
    return (__int64)v9;
  }
}
