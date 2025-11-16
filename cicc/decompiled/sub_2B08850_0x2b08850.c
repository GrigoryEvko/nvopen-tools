// Function: sub_2B08850
// Address: 0x2b08850
//
__int64 __fastcall sub_2B08850(__int64 a1, _DWORD *a2, int *a3)
{
  __int64 v5; // r12
  int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // rcx
  char v11; // di

  v5 = sub_22077B0(0x28u);
  v6 = *a3;
  *(_DWORD *)(v5 + 32) = *a2;
  *(_DWORD *)(v5 + 36) = v6;
  v7 = sub_2B086F0(a1, v5 + 32);
  v9 = v7;
  if ( v8 )
  {
    v10 = (_QWORD *)(a1 + 8);
    v11 = 1;
    if ( !v7 && (_QWORD *)v8 != v10 )
      v11 = v6 < *(_DWORD *)(v8 + 36);
    sub_220F040(v11, v5, (_QWORD *)v8, v10);
    ++*(_QWORD *)(a1 + 40);
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5);
    return v9;
  }
}
