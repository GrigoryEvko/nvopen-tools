// Function: sub_2A45970
// Address: 0x2a45970
//
void __fastcall sub_2A45970(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // r15
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r14
  unsigned __int64 *v11; // r12
  __int64 v12; // rbx
  void *v13; // rdi
  __int64 v14; // r9
  const void *v15; // rsi
  size_t v16; // rdx
  unsigned __int64 *v17; // rbx
  int v18; // ebx
  __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+0h] [rbp-50h]
  int v21; // [rsp+Ch] [rbp-44h]
  int v22; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (unsigned __int64 *)(a1 + 16);
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x30u, v23, a6);
  v9 = *(_QWORD *)a1;
  v10 = v8;
  v11 = (unsigned __int64 *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 == v11 )
    goto LABEL_16;
  v12 = v8;
  do
  {
    while ( 1 )
    {
      if ( !v12 )
        goto LABEL_3;
      v13 = (void *)(v12 + 16);
      *(_DWORD *)(v12 + 8) = 0;
      *(_QWORD *)v12 = v12 + 16;
      *(_DWORD *)(v12 + 12) = 4;
      v14 = *(unsigned int *)(v9 + 8);
      if ( !(_DWORD)v14 || v9 == v12 )
        goto LABEL_3;
      v15 = (const void *)(v9 + 16);
      if ( *(_QWORD *)v9 == v9 + 16 )
        break;
      *(_QWORD *)v12 = *(_QWORD *)v9;
      *(_DWORD *)(v12 + 8) = *(_DWORD *)(v9 + 8);
      *(_DWORD *)(v12 + 12) = *(_DWORD *)(v9 + 12);
      *(_QWORD *)v9 = v15;
      *(_DWORD *)(v9 + 12) = 0;
      *(_DWORD *)(v9 + 8) = 0;
LABEL_3:
      v9 += 48;
      v12 += 48;
      if ( v11 == (unsigned __int64 *)v9 )
        goto LABEL_11;
    }
    v16 = 8LL * (unsigned int)v14;
    if ( (unsigned int)v14 <= 4 )
      goto LABEL_9;
    v20 = v9;
    v22 = *(_DWORD *)(v9 + 8);
    sub_C8D5F0(v12, (const void *)(v12 + 16), (unsigned int)v14, 8u, v9, v14);
    v9 = v20;
    v13 = *(void **)v12;
    LODWORD(v14) = v22;
    v15 = *(const void **)v20;
    v16 = 8LL * *(unsigned int *)(v20 + 8);
    if ( v16 )
    {
LABEL_9:
      v19 = v9;
      v21 = v14;
      memcpy(v13, v15, v16);
      v9 = v19;
      LODWORD(v14) = v21;
    }
    *(_DWORD *)(v12 + 8) = v14;
    v9 += 48;
    v12 += 48;
    *(_DWORD *)(v9 - 40) = 0;
  }
  while ( v11 != (unsigned __int64 *)v9 );
LABEL_11:
  v17 = *(unsigned __int64 **)a1;
  v11 = (unsigned __int64 *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v11 )
  {
    do
    {
      v11 -= 6;
      if ( (unsigned __int64 *)*v11 != v11 + 2 )
        _libc_free(*v11);
    }
    while ( v11 != v17 );
    v11 = *(unsigned __int64 **)a1;
  }
LABEL_16:
  v18 = v23[0];
  if ( v6 != v11 )
    _libc_free((unsigned __int64)v11);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v18;
}
