// Function: sub_2C0FB70
// Address: 0x2c0fb70
//
void __fastcall sub_2C0FB70(__int64 a1)
{
  _QWORD *v2; // rdi
  __int64 *v3; // r14
  unsigned __int64 v4; // r12
  __int64 v5; // r15
  _QWORD *v6; // rdi
  __int64 v7; // rsi
  _QWORD *v8; // rax
  int v9; // r8d
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 **v12; // r12
  __int64 v13; // r13
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // r12
  __int64 *v16; // rdi
  __int64 v17; // rax
  __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = (_QWORD *)(a1 + 48);
  *(v2 - 6) = &unk_4A23AA8;
  *(v2 - 11) = &unk_4A23A70;
  sub_9C6650(v2);
  v3 = *(__int64 **)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A23170;
  v4 = (unsigned __int64)&v3[*(unsigned int *)(a1 + 16)];
  if ( v3 != (__int64 *)v4 )
  {
    do
    {
      v5 = *v3;
      v18[0] = a1;
      v6 = *(_QWORD **)(v5 + 16);
      v7 = (__int64)&v6[*(unsigned int *)(v5 + 24)];
      v8 = sub_2C0D780(v6, v7, v18);
      if ( (_QWORD *)v7 != v8 )
      {
        if ( (_QWORD *)v7 != v8 + 1 )
        {
          memmove(v8, v8 + 1, v7 - (_QWORD)(v8 + 1));
          v9 = *(_DWORD *)(v5 + 24);
        }
        *(_DWORD *)(v5 + 24) = v9 - 1;
      }
      ++v3;
    }
    while ( (__int64 *)v4 != v3 );
    v4 = *(_QWORD *)(a1 + 8);
  }
  if ( v4 != a1 + 24 )
    _libc_free(v4);
  *(_QWORD *)(a1 - 40) = &unk_4A231A8;
  v10 = *(_QWORD *)(a1 - 24);
  v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v10 & 4) != 0 )
  {
    v12 = *(__int64 ***)v11;
    v13 = *(_QWORD *)v11 + 8LL * *(unsigned int *)(v11 + 8);
  }
  else
  {
    v12 = (__int64 **)(a1 - 24);
    v13 = a1 - 16;
    if ( !v11 )
      goto LABEL_12;
  }
  if ( (__int64 **)v13 != v12 )
  {
    do
    {
      v16 = *v12++;
      v17 = *v16;
      v16[6] = 0;
      (*(void (**)(void))(v17 + 8))();
    }
    while ( (__int64 **)v13 != v12 );
    v10 = *(_QWORD *)(a1 - 24);
    if ( !v10 )
      return;
    goto LABEL_13;
  }
LABEL_12:
  if ( !v10 )
    return;
LABEL_13:
  if ( (v10 & 4) != 0 )
  {
    v14 = (unsigned __int64 *)(v10 & 0xFFFFFFFFFFFFFFF8LL);
    v15 = (unsigned __int64)v14;
    if ( v14 )
    {
      if ( (unsigned __int64 *)*v14 != v14 + 2 )
        _libc_free(*v14);
      j_j___libc_free_0(v15);
    }
  }
}
