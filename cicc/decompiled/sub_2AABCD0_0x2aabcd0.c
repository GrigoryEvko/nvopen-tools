// Function: sub_2AABCD0
// Address: 0x2aabcd0
//
void __fastcall sub_2AABCD0(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 *v4; // r14
  unsigned __int64 v5; // r12
  __int64 v6; // r15
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
  __int64 v19[7]; // [rsp+8h] [rbp-38h] BYREF

  *(_QWORD *)(a1 - 40) = &unk_4A23718;
  *(_QWORD *)(a1 + 56) = &unk_4A23790;
  v2 = a1 + 144;
  *(_QWORD *)a1 = &unk_4A23758;
  v3 = *(_QWORD *)(a1 + 128);
  if ( v3 != v2 )
    j_j___libc_free_0(v3);
  *(_QWORD *)(a1 - 40) = &unk_4A231C8;
  *(_QWORD *)a1 = &unk_4A23200;
  *(_QWORD *)(a1 + 56) = &unk_4A23238;
  sub_2BF1E70(a1 + 56);
  *(_QWORD *)a1 = &unk_4A23AA8;
  *(_QWORD *)(a1 - 40) = &unk_4A23A70;
  sub_9C6650((_QWORD *)(a1 + 48));
  v4 = *(__int64 **)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A23170;
  v5 = (unsigned __int64)&v4[*(unsigned int *)(a1 + 16)];
  if ( v4 != (__int64 *)v5 )
  {
    do
    {
      v6 = *v4;
      v19[0] = a1;
      v7 = *(_QWORD **)(v6 + 16);
      v8 = (__int64)&v7[*(unsigned int *)(v6 + 24)];
      v9 = sub_2AA89B0(v7, v8, v19);
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
      goto LABEL_14;
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
    if ( !v11 )
      return;
    goto LABEL_15;
  }
LABEL_14:
  if ( !v11 )
    return;
LABEL_15:
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
