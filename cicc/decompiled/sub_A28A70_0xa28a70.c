// Function: sub_A28A70
// Address: 0xa28a70
//
__int64 __fastcall sub_A28A70(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r13
  _QWORD *v11; // r15
  __int64 v12; // rdi
  __int64 v14; // rdi

  v5 = sub_22077B0(72);
  v6 = *a3;
  *(_QWORD *)(v5 + 32) = v5 + 48;
  sub_A15D40((__int64 *)(v5 + 32), *(_BYTE **)v6, *(_QWORD *)v6 + *(_QWORD *)(v6 + 8));
  *(_DWORD *)(v5 + 64) = 0;
  v7 = sub_A288A0(a1, a2, v5 + 32);
  v9 = v7;
  if ( v8 )
  {
    v10 = v8;
    v11 = a1 + 1;
    v12 = 1;
    if ( !v7 && (_QWORD *)v8 != v11 )
      v12 = (unsigned int)sub_A15B80(
                            *(const void **)(v5 + 32),
                            *(_QWORD *)(v5 + 40),
                            *(const void **)(v8 + 32),
                            *(_QWORD *)(v8 + 40)) >> 31;
    sub_220F040(v12, v5, v10, a1 + 1);
    ++a1[5];
    return v5;
  }
  else
  {
    v14 = *(_QWORD *)(v5 + 32);
    if ( v5 + 48 != v14 )
      j_j___libc_free_0(v14, *(_QWORD *)(v5 + 48) + 1LL);
    j_j___libc_free_0(v5, 72);
    return v9;
  }
}
