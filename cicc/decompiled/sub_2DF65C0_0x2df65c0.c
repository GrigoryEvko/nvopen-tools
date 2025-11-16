// Function: sub_2DF65C0
// Address: 0x2df65c0
//
__int64 __fastcall sub_2DF65C0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rax
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // r14
  __int64 v11; // rbx
  __int64 v12; // rax
  char v13; // di
  int v14; // edi
  __int64 v15; // rax
  unsigned __int64 v16; // r8
  void *v17; // rdi
  size_t v18; // rdx
  const void *v19; // rsi
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  int v22; // ebx
  __int64 v24; // [rsp+8h] [rbp-48h]
  unsigned __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v25, a6);
  v24 = v8;
  v9 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = *(_QWORD *)a1;
    v11 = v8;
    do
    {
      while ( 1 )
      {
        if ( v11 )
        {
          v12 = *(_QWORD *)v10;
          *(_QWORD *)(v11 + 8) = 0;
          *(_QWORD *)v11 = v12;
          v13 = *(_BYTE *)(v10 + 16);
          *(_BYTE *)(v11 + 16) = v13;
          *(_QWORD *)(v11 + 24) = *(_QWORD *)(v10 + 24);
          v14 = v13 & 0x3F;
          if ( v14 )
          {
            v15 = sub_2207820(4LL * (unsigned __int8)v14);
            v16 = *(_QWORD *)(v11 + 8);
            v17 = (void *)v15;
            *(_QWORD *)(v11 + 8) = v15;
            if ( v16 )
            {
              j_j___libc_free_0_0(v16);
              v17 = *(void **)(v11 + 8);
            }
            v18 = 4LL * (*(_BYTE *)(v10 + 16) & 0x3F);
            if ( v18 )
              break;
          }
        }
        v10 += 32LL;
        v11 += 32;
        if ( v9 == v10 )
          goto LABEL_10;
      }
      v19 = *(const void **)(v10 + 8);
      v10 += 32LL;
      v11 += 32;
      memmove(v17, v19, v18);
    }
    while ( v9 != v10 );
LABEL_10:
    v20 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v21 = *(_QWORD *)(v9 - 24);
        v9 -= 32LL;
        if ( v21 )
          j_j___libc_free_0_0(v21);
      }
      while ( v9 != v20 );
      v9 = *(_QWORD *)a1;
    }
  }
  v22 = v25[0];
  if ( v6 != v9 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v22;
  *(_QWORD *)a1 = v24;
  return v24;
}
