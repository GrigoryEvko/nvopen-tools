// Function: sub_359C370
// Address: 0x359c370
//
void __fastcall sub_359C370(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rcx
  unsigned __int64 v12; // r14
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rsi
  unsigned __int64 v16; // r15
  __int64 v17; // rsi
  __int64 v18; // rdi
  int v19; // r15d
  unsigned __int64 v20[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v20, a6);
  v9 = *(_QWORD *)a1;
  v10 = v8;
  v11 = 32LL * *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1 + v11;
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = v8 + v11;
    do
    {
      if ( v8 )
      {
        *(_DWORD *)(v8 + 24) = 0;
        *(_QWORD *)(v8 + 8) = 0;
        *(_DWORD *)(v8 + 16) = 0;
        *(_DWORD *)(v8 + 20) = 0;
        *(_QWORD *)v8 = 1;
        v14 = *(_QWORD *)(v9 + 8);
        ++*(_QWORD *)v9;
        v15 = *(_QWORD *)(v8 + 8);
        *(_QWORD *)(v8 + 8) = v14;
        LODWORD(v14) = *(_DWORD *)(v9 + 16);
        *(_QWORD *)(v9 + 8) = v15;
        LODWORD(v15) = *(_DWORD *)(v8 + 16);
        *(_DWORD *)(v8 + 16) = v14;
        LODWORD(v14) = *(_DWORD *)(v9 + 20);
        *(_DWORD *)(v9 + 16) = v15;
        LODWORD(v15) = *(_DWORD *)(v8 + 20);
        *(_DWORD *)(v8 + 20) = v14;
        LODWORD(v14) = *(_DWORD *)(v9 + 24);
        *(_DWORD *)(v9 + 20) = v15;
        LODWORD(v15) = *(_DWORD *)(v8 + 24);
        *(_DWORD *)(v8 + 24) = v14;
        *(_DWORD *)(v9 + 24) = v15;
      }
      v8 += 32;
      v9 += 32LL;
    }
    while ( v8 != v13 );
    v16 = *(_QWORD *)a1;
    v12 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v12 )
    {
      do
      {
        v17 = *(unsigned int *)(v12 - 8);
        v18 = *(_QWORD *)(v12 - 24);
        v12 -= 32LL;
        sub_C7D6A0(v18, 8 * v17, 4);
      }
      while ( v12 != v16 );
      v12 = *(_QWORD *)a1;
    }
  }
  v19 = v20[0];
  if ( v6 != v12 )
    _libc_free(v12);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v19;
}
