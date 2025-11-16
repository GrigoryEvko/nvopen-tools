// Function: sub_2AAB5E0
// Address: 0x2aab5e0
//
void __fastcall sub_2AAB5E0(_QWORD *a1)
{
  __int64 *v2; // r15
  unsigned __int64 v3; // r12
  __int64 v4; // r13
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  _QWORD *v7; // rax
  int v8; // r8d
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 **v11; // r12
  __int64 v12; // r13
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // r12
  __int64 *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  *(a1 - 12) = &unk_4A231C8;
  *(a1 - 7) = &unk_4A23200;
  *a1 = &unk_4A23238;
  sub_2BF1E70();
  *(a1 - 7) = &unk_4A23AA8;
  *(a1 - 12) = &unk_4A23A70;
  sub_9C6650(a1 - 1);
  v2 = (__int64 *)*(a1 - 6);
  v17 = (__int64)(a1 - 7);
  *(a1 - 7) = &unk_4A23170;
  v3 = (unsigned __int64)&v2[*((unsigned int *)a1 - 10)];
  if ( v2 != (__int64 *)v3 )
  {
    do
    {
      v4 = *v2;
      v18[0] = v17;
      v5 = *(_QWORD **)(v4 + 16);
      v6 = (__int64)&v5[*(unsigned int *)(v4 + 24)];
      v7 = sub_2AA89B0(v5, v6, v18);
      if ( (_QWORD *)v6 != v7 )
      {
        if ( (_QWORD *)v6 != v7 + 1 )
        {
          memmove(v7, v7 + 1, v6 - (_QWORD)(v7 + 1));
          v8 = *(_DWORD *)(v4 + 24);
        }
        *(_DWORD *)(v4 + 24) = v8 - 1;
      }
      ++v2;
    }
    while ( (__int64 *)v3 != v2 );
    v3 = *(a1 - 6);
  }
  if ( (_QWORD *)v3 != a1 - 4 )
    _libc_free(v3);
  *(a1 - 12) = &unk_4A231A8;
  v9 = *(a1 - 10);
  v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v9 & 4) != 0 )
  {
    v11 = *(__int64 ***)v10;
    v12 = *(_QWORD *)v10 + 8LL * *(unsigned int *)(v10 + 8);
  }
  else
  {
    v11 = (__int64 **)(a1 - 10);
    v12 = (__int64)(a1 - 9);
    if ( !v10 )
      goto LABEL_12;
  }
  if ( (__int64 **)v12 != v11 )
  {
    do
    {
      v15 = *v11++;
      v16 = *v15;
      v15[6] = 0;
      (*(void (**)(void))(v16 + 8))();
    }
    while ( (__int64 **)v12 != v11 );
    v9 = *(a1 - 10);
    if ( !v9 )
      return;
    goto LABEL_13;
  }
LABEL_12:
  if ( !v9 )
    return;
LABEL_13:
  if ( (v9 & 4) != 0 )
  {
    v13 = (unsigned __int64 *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
    v14 = (unsigned __int64)v13;
    if ( v13 )
    {
      if ( (unsigned __int64 *)*v13 != v13 + 2 )
        _libc_free(*v13);
      j_j___libc_free_0(v14);
    }
  }
}
