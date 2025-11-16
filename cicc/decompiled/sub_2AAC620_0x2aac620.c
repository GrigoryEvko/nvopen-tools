// Function: sub_2AAC620
// Address: 0x2aac620
//
void __fastcall sub_2AAC620(_QWORD *a1)
{
  __int64 *v2; // rcx
  unsigned __int64 v3; // r12
  __int64 *v4; // r15
  __int64 v5; // r13
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
  unsigned __int64 v18; // [rsp+0h] [rbp-50h]
  __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v18 = (unsigned __int64)(a1 - 12);
  *(a1 - 12) = &unk_4A231C8;
  *(a1 - 7) = &unk_4A23200;
  *a1 = &unk_4A23238;
  sub_2BF1E70(a1);
  *(a1 - 7) = &unk_4A23AA8;
  *(a1 - 12) = &unk_4A23A70;
  sub_9C6650(a1 - 1);
  v2 = (__int64 *)*(a1 - 6);
  v19 = (__int64)(a1 - 7);
  *(a1 - 7) = &unk_4A23170;
  v3 = (unsigned __int64)&v2[*((unsigned int *)a1 - 10)];
  if ( v2 != (__int64 *)v3 )
  {
    v4 = v2;
    do
    {
      v5 = *v4;
      v20[0] = v19;
      v6 = *(_QWORD **)(v5 + 16);
      v7 = (__int64)&v6[*(unsigned int *)(v5 + 24)];
      v8 = sub_2AA89B0(v6, v7, v20);
      if ( (_QWORD *)v7 != v8 )
      {
        if ( (_QWORD *)v7 != v8 + 1 )
        {
          memmove(v8, v8 + 1, v7 - (_QWORD)(v8 + 1));
          v9 = *(_DWORD *)(v5 + 24);
        }
        *(_DWORD *)(v5 + 24) = v9 - 1;
      }
      ++v4;
    }
    while ( (__int64 *)v3 != v4 );
    v3 = *(a1 - 6);
  }
  if ( (_QWORD *)v3 != a1 - 4 )
    _libc_free(v3);
  *(a1 - 12) = &unk_4A231A8;
  v10 = *(a1 - 10);
  v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v10 & 4) != 0 )
  {
    v12 = *(__int64 ***)v11;
    v13 = *(_QWORD *)v11 + 8LL * *(unsigned int *)(v11 + 8);
  }
  else
  {
    v12 = (__int64 **)(a1 - 10);
    v13 = (__int64)(a1 - 9);
    if ( !v11 )
      goto LABEL_13;
  }
  if ( (__int64 **)v13 != v12 )
  {
    do
    {
      v16 = *v12++;
      v17 = *v16;
      v16[6] = 0;
      (*(void (__fastcall **)(__int64 *))(v17 + 8))(v16);
    }
    while ( (__int64 **)v13 != v12 );
    v10 = *(a1 - 10);
  }
LABEL_13:
  if ( v10 )
  {
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
  j_j___libc_free_0(v18);
}
