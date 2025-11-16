// Function: sub_8565C0
// Address: 0x8565c0
//
void __fastcall sub_8565C0(_QWORD *a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  _QWORD *v4; // rdi
  __int64 v5; // rax
  void *v6; // rdi
  char *v7; // r13
  char *v8; // rdi
  __int64 v9; // rsi

  v2 = (_QWORD *)a1[2];
  while ( v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)*v2;
    v4 = (_QWORD *)v3[1];
    if ( v4 != v3 + 3 )
      j_j___libc_free_0(v4, v3[3] + 1LL);
    j_j___libc_free_0(v3, 48);
  }
  v5 = a1[1];
  v6 = (void *)*a1;
  v7 = (char *)(a1 + 6);
  memset(v6, 0, 8 * v5);
  v8 = (char *)*((_QWORD *)v7 - 6);
  v9 = *((_QWORD *)v7 - 5);
  *((_QWORD *)v7 - 3) = 0;
  *((_QWORD *)v7 - 4) = 0;
  if ( v8 != v7 )
    j_j___libc_free_0(v8, 8 * v9);
}
