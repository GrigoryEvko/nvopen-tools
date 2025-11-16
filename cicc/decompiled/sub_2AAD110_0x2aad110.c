// Function: sub_2AAD110
// Address: 0x2aad110
//
void __fastcall sub_2AAD110(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi
  __int64 *v4; // rcx
  unsigned __int64 v5; // r12
  __int64 *v6; // r15
  __int64 v7; // r13
  _QWORD *v8; // rdi
  __int64 v9; // rsi
  _QWORD *v10; // rax
  int v11; // r8d
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 **v14; // r12
  __int64 v15; // r13
  unsigned __int64 *v16; // rax
  unsigned __int64 v17; // r12
  __int64 *v18; // rdi
  __int64 v19; // rax
  unsigned __int64 v20; // [rsp+0h] [rbp-50h]
  __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v20 = (unsigned __int64)(a1 - 12);
  *(a1 - 12) = &unk_4A23718;
  *a1 = &unk_4A23790;
  v2 = a1 + 11;
  *(a1 - 7) = &unk_4A23758;
  v3 = a1[9];
  if ( (_QWORD *)v3 != v2 )
    j_j___libc_free_0(v3);
  *(a1 - 12) = &unk_4A231C8;
  *(a1 - 7) = &unk_4A23200;
  *a1 = &unk_4A23238;
  sub_2BF1E70(a1);
  *(a1 - 7) = &unk_4A23AA8;
  *(a1 - 12) = &unk_4A23A70;
  sub_9C6650(a1 - 1);
  v4 = (__int64 *)*(a1 - 6);
  *(a1 - 7) = &unk_4A23170;
  v5 = (unsigned __int64)&v4[*((unsigned int *)a1 - 10)];
  if ( v4 != (__int64 *)v5 )
  {
    v6 = v4;
    do
    {
      v7 = *v6;
      v21[0] = (__int64)(a1 - 7);
      v8 = *(_QWORD **)(v7 + 16);
      v9 = (__int64)&v8[*(unsigned int *)(v7 + 24)];
      v10 = sub_2AA89B0(v8, v9, v21);
      if ( (_QWORD *)v9 != v10 )
      {
        if ( (_QWORD *)v9 != v10 + 1 )
        {
          memmove(v10, v10 + 1, v9 - (_QWORD)(v10 + 1));
          v11 = *(_DWORD *)(v7 + 24);
        }
        *(_DWORD *)(v7 + 24) = v11 - 1;
      }
      ++v6;
    }
    while ( (__int64 *)v5 != v6 );
    v5 = *(a1 - 6);
  }
  if ( (_QWORD *)v5 != a1 - 4 )
    _libc_free(v5);
  *(a1 - 12) = &unk_4A231A8;
  v12 = *(a1 - 10);
  v13 = v12 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v12 & 4) != 0 )
  {
    v14 = *(__int64 ***)v13;
    v15 = *(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8);
  }
  else
  {
    v14 = (__int64 **)(a1 - 10);
    v15 = (__int64)(a1 - 9);
    if ( !v13 )
      goto LABEL_15;
  }
  if ( (__int64 **)v15 != v14 )
  {
    do
    {
      v18 = *v14++;
      v19 = *v18;
      v18[6] = 0;
      (*(void (__fastcall **)(__int64 *))(v19 + 8))(v18);
    }
    while ( (__int64 **)v15 != v14 );
    v12 = *(a1 - 10);
  }
LABEL_15:
  if ( v12 )
  {
    if ( (v12 & 4) != 0 )
    {
      v16 = (unsigned __int64 *)(v12 & 0xFFFFFFFFFFFFFFF8LL);
      v17 = (unsigned __int64)v16;
      if ( v16 )
      {
        if ( (unsigned __int64 *)*v16 != v16 + 2 )
          _libc_free(*v16);
        j_j___libc_free_0(v17);
      }
    }
  }
  j_j___libc_free_0(v20);
}
