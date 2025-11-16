// Function: sub_2C10320
// Address: 0x2c10320
//
void __fastcall sub_2C10320(__int64 a1)
{
  unsigned __int64 v1; // rax
  _QWORD *v3; // rdi
  __int64 *v4; // r15
  unsigned __int64 v5; // r12
  __int64 v6; // r14
  _QWORD *v7; // rdi
  __int64 v8; // rsi
  _QWORD *v9; // rax
  int v10; // r8d
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 **v13; // r12
  __int64 v14; // r13
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // r12
  __int64 *v17; // rdi
  __int64 v18; // rax
  unsigned __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = a1 - 40;
  v3 = (_QWORD *)(a1 + 48);
  v19 = v1;
  *(v3 - 6) = &unk_4A23AA8;
  *(v3 - 11) = &unk_4A23A70;
  sub_9C6650(v3);
  v4 = *(__int64 **)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A23170;
  v5 = (unsigned __int64)&v4[*(unsigned int *)(a1 + 16)];
  if ( v4 != (__int64 *)v5 )
  {
    do
    {
      v6 = *v4;
      v20[0] = a1;
      v7 = *(_QWORD **)(v6 + 16);
      v8 = (__int64)&v7[*(unsigned int *)(v6 + 24)];
      v9 = sub_2C0D780(v7, v8, v20);
      if ( (_QWORD *)v8 != v9 )
      {
        if ( (_QWORD *)v8 != v9 + 1 )
        {
          memmove(v9, v9 + 1, v8 - (_QWORD)(v9 + 1));
          v10 = *(_DWORD *)(v6 + 24);
        }
        *(_DWORD *)(v6 + 24) = v10 - 1;
      }
      ++v4;
    }
    while ( (__int64 *)v5 != v4 );
    v5 = *(_QWORD *)(a1 + 8);
  }
  if ( v5 != a1 + 24 )
    _libc_free(v5);
  *(_QWORD *)(a1 - 40) = &unk_4A231A8;
  v11 = *(_QWORD *)(a1 - 24);
  v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v11 & 4) != 0 )
  {
    v13 = *(__int64 ***)v12;
    v14 = *(_QWORD *)v12 + 8LL * *(unsigned int *)(v12 + 8);
  }
  else
  {
    v13 = (__int64 **)(a1 - 24);
    v14 = a1 - 16;
    if ( !v12 )
      goto LABEL_12;
  }
  if ( (__int64 **)v14 != v13 )
  {
    do
    {
      v17 = *v13++;
      v18 = *v17;
      v17[6] = 0;
      (*(void (**)(void))(v18 + 8))();
    }
    while ( (__int64 **)v14 != v13 );
    v11 = *(_QWORD *)(a1 - 24);
  }
LABEL_12:
  if ( v11 )
  {
    if ( (v11 & 4) != 0 )
    {
      v15 = (unsigned __int64 *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
      v16 = (unsigned __int64)v15;
      if ( v15 )
      {
        if ( (unsigned __int64 *)*v15 != v15 + 2 )
          _libc_free(*v15);
        j_j___libc_free_0(v16);
      }
    }
  }
  j_j___libc_free_0(v19);
}
