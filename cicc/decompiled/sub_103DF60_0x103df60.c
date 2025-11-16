// Function: sub_103DF60
// Address: 0x103df60
//
_QWORD *__fastcall sub_103DF60(_QWORD *a1)
{
  _QWORD *v1; // r12
  _QWORD *v3; // r13
  _QWORD *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r12
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rdi

  v1 = (_QWORD *)a1[42];
  if ( v1 )
    return v1;
  v3 = (_QWORD *)a1[41];
  if ( !v3 )
  {
    v7 = a1[1];
    v8 = (_QWORD *)sub_22077B0(2400);
    v3 = v8;
    if ( v8 )
    {
      *v8 = a1;
      v9 = v8 + 7;
      *(v9 - 6) = v7;
      v3[5] = v9;
      v3[6] = 0x2000000000LL;
      v3[295] = 0;
      v3[296] = 0;
      v3[297] = 0;
      *((_DWORD *)v3 + 596) = 0;
      v3[299] = a1;
    }
    v10 = a1[41];
    a1[41] = v3;
    if ( v10 )
    {
      v11 = 56LL * *(unsigned int *)(v10 + 2384);
      sub_C7D6A0(*(_QWORD *)(v10 + 2368), v11, 8);
      v12 = *(_QWORD *)(v10 + 40);
      if ( v12 != v10 + 56 )
        _libc_free(v12, v11);
      j_j___libc_free_0(v10, 2400);
      v3 = (_QWORD *)a1[41];
    }
  }
  v5 = (_QWORD *)sub_22077B0(24);
  v1 = v5;
  if ( v5 )
  {
    sub_103DDC0(v5, (__int64)a1);
    v1[2] = v3;
    *v1 = &unk_49E5AA8;
  }
  v6 = a1[42];
  a1[42] = v1;
  if ( !v6 )
    return v1;
  j_j___libc_free_0(v6, 24);
  return (_QWORD *)a1[42];
}
