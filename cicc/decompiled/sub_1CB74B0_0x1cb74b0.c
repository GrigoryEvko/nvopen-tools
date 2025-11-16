// Function: sub_1CB74B0
// Address: 0x1cb74b0
//
_QWORD *__fastcall sub_1CB74B0(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // r14
  __int64 v5; // r12
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r13
  __int64 v9; // rcx
  _BOOL8 v10; // rdi

  v3 = sub_22077B0(48);
  v4 = *a2;
  v5 = v3;
  *(_QWORD *)(v3 + 32) = *a2;
  *(_DWORD *)(v3 + 40) = *((_DWORD *)a2 + 2);
  v6 = sub_1C70290(a1, (unsigned __int64 *)(v3 + 32));
  v8 = v6;
  if ( v7 )
  {
    v9 = a1 + 8;
    v10 = 1;
    if ( !v6 && v7 != v9 )
      v10 = v4 < *(_QWORD *)(v7 + 32);
    sub_220F040(v10, v5, v7, v9);
    ++*(_QWORD *)(a1 + 40);
    return (_QWORD *)v5;
  }
  else
  {
    j_j___libc_free_0(v5, 48);
    return v8;
  }
}
