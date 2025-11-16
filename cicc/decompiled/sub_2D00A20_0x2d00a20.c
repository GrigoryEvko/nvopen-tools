// Function: sub_2D00A20
// Address: 0x2d00a20
//
unsigned __int64 __fastcall sub_2D00A20(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // r12
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  _QWORD *v8; // r13
  _QWORD *v9; // rcx
  char v10; // di

  v3 = sub_22077B0(0x30u);
  v4 = *a2;
  v5 = v3;
  *(_QWORD *)(v3 + 32) = *a2;
  *(_DWORD *)(v3 + 40) = *((_DWORD *)a2 + 2);
  v6 = sub_2CBB810(a1, (unsigned __int64 *)(v3 + 32));
  v8 = v6;
  if ( v7 )
  {
    v9 = (_QWORD *)(a1 + 8);
    v10 = 1;
    if ( !v6 && v7 != v9 )
      v10 = v4 < v7[4];
    sub_220F040(v10, v5, v7, v9);
    ++*(_QWORD *)(a1 + 40);
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5);
    return (unsigned __int64)v8;
  }
}
