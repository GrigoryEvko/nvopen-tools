// Function: sub_35498F0
// Address: 0x35498f0
//
__int64 __fastcall sub_35498F0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v12; // r12
  __int64 v13; // rbx
  unsigned __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  int v20; // ebx
  __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x58u, v23, a6);
  v12 = *(_QWORD *)a1;
  v22 = v8;
  v13 = v8;
  v14 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v14 )
  {
    do
    {
      if ( v13 )
      {
        *(_DWORD *)(v13 + 24) = 0;
        *(_QWORD *)(v13 + 8) = 0;
        *(_DWORD *)(v13 + 16) = 0;
        *(_DWORD *)(v13 + 20) = 0;
        *(_QWORD *)v13 = 1;
        v15 = *(_QWORD *)(v12 + 8);
        ++*(_QWORD *)v12;
        v16 = *(_QWORD *)(v13 + 8);
        *(_QWORD *)(v13 + 8) = v15;
        LODWORD(v15) = *(_DWORD *)(v12 + 16);
        *(_QWORD *)(v12 + 8) = v16;
        LODWORD(v16) = *(_DWORD *)(v13 + 16);
        *(_DWORD *)(v13 + 16) = v15;
        LODWORD(v15) = *(_DWORD *)(v12 + 20);
        *(_DWORD *)(v12 + 16) = v16;
        LODWORD(v16) = *(_DWORD *)(v13 + 20);
        *(_DWORD *)(v13 + 20) = v15;
        v17 = *(unsigned int *)(v12 + 24);
        *(_DWORD *)(v12 + 20) = v16;
        LODWORD(v16) = *(_DWORD *)(v13 + 24);
        *(_DWORD *)(v13 + 24) = v17;
        *(_DWORD *)(v12 + 24) = v16;
        *(_QWORD *)(v13 + 32) = v13 + 48;
        *(_DWORD *)(v13 + 40) = 0;
        *(_DWORD *)(v13 + 44) = 0;
        if ( *(_DWORD *)(v12 + 40) )
          sub_353DE10(v13 + 32, (char **)(v12 + 32), v17, v9, v10, v11);
        *(_BYTE *)(v13 + 48) = *(_BYTE *)(v12 + 48);
        *(_DWORD *)(v13 + 52) = *(_DWORD *)(v12 + 52);
        *(_DWORD *)(v13 + 56) = *(_DWORD *)(v12 + 56);
        *(_DWORD *)(v13 + 60) = *(_DWORD *)(v12 + 60);
        *(_DWORD *)(v13 + 64) = *(_DWORD *)(v12 + 64);
        *(_QWORD *)(v13 + 72) = *(_QWORD *)(v12 + 72);
        *(_DWORD *)(v13 + 80) = *(_DWORD *)(v12 + 80);
      }
      v12 += 88LL;
      v13 += 88;
    }
    while ( v14 != v12 );
    v18 = *(_QWORD *)a1;
    v14 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v14 )
    {
      do
      {
        v14 -= 88LL;
        v19 = *(_QWORD *)(v14 + 32);
        if ( v19 != v14 + 48 )
          _libc_free(v19);
        sub_C7D6A0(*(_QWORD *)(v14 + 8), 8LL * *(unsigned int *)(v14 + 24), 8);
      }
      while ( v14 != v18 );
      v14 = *(_QWORD *)a1;
    }
  }
  v20 = v23[0];
  if ( v6 != v14 )
    _libc_free(v14);
  *(_DWORD *)(a1 + 12) = v20;
  *(_QWORD *)a1 = v22;
  return v22;
}
