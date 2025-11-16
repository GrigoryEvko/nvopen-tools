// Function: sub_E9E2A0
// Address: 0xe9e2a0
//
__int64 __fastcall sub_E9E2A0(_QWORD *a1, __int64 a2, unsigned int **a3)
{
  __int64 v5; // r12
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // rcx
  _BOOL8 v11; // rdi

  v5 = sub_22077B0(608);
  v6 = **a3;
  *(_DWORD *)(v5 + 32) = v6;
  memset((void *)(v5 + 40), 0, 0x238u);
  *(_QWORD *)(v5 + 160) = v5 + 176;
  *(_QWORD *)(v5 + 48) = v5 + 64;
  *(_QWORD *)(v5 + 56) = 0x300000000LL;
  *(_QWORD *)(v5 + 168) = 0x300000000LL;
  *(_QWORD *)(v5 + 432) = 0x1000000000LL;
  *(_QWORD *)(v5 + 440) = v5 + 456;
  *(_QWORD *)(v5 + 472) = v5 + 488;
  *(_BYTE *)(v5 + 553) = 1;
  *(_QWORD *)(v5 + 592) = v5 + 608;
  v7 = sub_E55F30(a1, a2, (unsigned int *)(v5 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && (_QWORD *)v8 != v10 )
      v11 = v6 < *(_DWORD *)(v8 + 32);
    sub_220F040(v11, v5, v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    sub_E56030(v5 + 40, a2);
    j_j___libc_free_0(v5, 608);
    return v9;
  }
}
