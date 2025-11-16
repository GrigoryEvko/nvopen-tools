// Function: sub_31B0D60
// Address: 0x31b0d60
//
void __fastcall sub_31B0D60(__int64 a1)
{
  __int64 v1; // r14
  unsigned __int64 v3; // r13
  _QWORD *v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // r14
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi

  v1 = *(_QWORD *)(a1 + 104);
  *(_QWORD *)a1 = &unk_4A348B0;
  v3 = v1 + 8LL * *(unsigned int *)(a1 + 112);
  if ( v1 != v3 )
  {
    do
    {
      v4 = *(_QWORD **)(v3 - 8);
      v3 -= 8LL;
      if ( v4 )
      {
        v5 = v4[17];
        if ( (_QWORD *)v5 != v4 + 19 )
          _libc_free(v5);
        v6 = v4[8];
        if ( (_QWORD *)v6 != v4 + 10 )
          _libc_free(v6);
        v7 = v4[2];
        if ( (_QWORD *)v7 != v4 + 4 )
          _libc_free(v7);
        j_j___libc_free_0((unsigned __int64)v4);
      }
    }
    while ( v1 != v3 );
    v3 = *(_QWORD *)(a1 + 104);
  }
  if ( v3 != a1 + 120 )
    _libc_free(v3);
  v8 = *(_QWORD *)(a1 + 88);
  if ( v8 )
  {
    v9 = *(unsigned int *)(v8 + 56);
    if ( (_DWORD)v9 )
    {
      v10 = *(_QWORD *)(v8 + 40);
      v11 = v10 + 40 * v9;
      do
      {
        if ( *(_QWORD *)v10 != -4096 && *(_QWORD *)v10 != -8192 )
          sub_C7D6A0(*(_QWORD *)(v10 + 16), 16LL * *(unsigned int *)(v10 + 32), 8);
        v10 += 40;
      }
      while ( v11 != v10 );
      v9 = *(unsigned int *)(v8 + 56);
    }
    sub_C7D6A0(*(_QWORD *)(v8 + 40), 40 * v9, 8);
    sub_C7D6A0(*(_QWORD *)(v8 + 8), 16LL * *(unsigned int *)(v8 + 24), 8);
    j_j___libc_free_0(v8);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 64), 8LL * *(unsigned int *)(a1 + 80), 8);
  v12 = *(_QWORD *)(a1 + 48);
  if ( v12 )
    sub_31B0BF0(v12);
  v13 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A23850;
  if ( v13 != a1 + 24 )
    j_j___libc_free_0(v13);
}
