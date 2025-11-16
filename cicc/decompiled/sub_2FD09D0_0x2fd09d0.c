// Function: sub_2FD09D0
// Address: 0x2fd09d0
//
__int64 __fastcall sub_2FD09D0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // rax
  unsigned __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r8
  unsigned __int64 v12; // r13
  void *v13; // rdi
  __int64 v14; // r9
  const void *v15; // rsi
  size_t v16; // rdx
  unsigned __int64 *v17; // rbx
  int v18; // ebx
  int v20; // [rsp+4h] [rbp-4Ch]
  int v21; // [rsp+4h] [rbp-4Ch]
  __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x50u, v23, a6);
  v9 = *(_QWORD *)a1;
  v22 = v8;
  v10 = v8;
  v11 = 80LL * *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1 + v11;
  if ( *(_QWORD *)a1 == v12 )
    goto LABEL_16;
  do
  {
    while ( 1 )
    {
      if ( !v10 )
        goto LABEL_3;
      v13 = (void *)(v10 + 16);
      *(_DWORD *)(v10 + 8) = 0;
      *(_QWORD *)v10 = v10 + 16;
      *(_DWORD *)(v10 + 12) = 8;
      v14 = *(unsigned int *)(v9 + 8);
      if ( !(_DWORD)v14 || v9 == v10 )
        goto LABEL_3;
      v15 = (const void *)(v9 + 16);
      if ( *(_QWORD *)v9 == v9 + 16 )
        break;
      *(_QWORD *)v10 = *(_QWORD *)v9;
      *(_DWORD *)(v10 + 8) = *(_DWORD *)(v9 + 8);
      *(_DWORD *)(v10 + 12) = *(_DWORD *)(v9 + 12);
      *(_QWORD *)v9 = v15;
      *(_DWORD *)(v9 + 12) = 0;
      *(_DWORD *)(v9 + 8) = 0;
LABEL_3:
      v9 += 80LL;
      v10 += 80;
      if ( v12 == v9 )
        goto LABEL_11;
    }
    v16 = 8LL * (unsigned int)v14;
    if ( (unsigned int)v14 <= 8
      || (v21 = *(_DWORD *)(v9 + 8),
          sub_C8D5F0(v10, (const void *)(v10 + 16), (unsigned int)v14, 8u, v11, v14),
          v13 = *(void **)v10,
          v15 = *(const void **)v9,
          LODWORD(v14) = v21,
          (v16 = 8LL * *(unsigned int *)(v9 + 8)) != 0) )
    {
      v20 = v14;
      memcpy(v13, v15, v16);
      LODWORD(v14) = v20;
    }
    *(_DWORD *)(v10 + 8) = v14;
    v9 += 80LL;
    v10 += 80;
    *(_DWORD *)(v9 - 72) = 0;
  }
  while ( v12 != v9 );
LABEL_11:
  v12 = *(_QWORD *)a1;
  v17 = (unsigned __int64 *)(*(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v17 )
  {
    do
    {
      v17 -= 10;
      if ( (unsigned __int64 *)*v17 != v17 + 2 )
        _libc_free(*v17);
    }
    while ( (unsigned __int64 *)v12 != v17 );
    v12 = *(_QWORD *)a1;
  }
LABEL_16:
  v18 = v23[0];
  if ( v6 != v12 )
    _libc_free(v12);
  *(_DWORD *)(a1 + 12) = v18;
  *(_QWORD *)a1 = v22;
  return v22;
}
