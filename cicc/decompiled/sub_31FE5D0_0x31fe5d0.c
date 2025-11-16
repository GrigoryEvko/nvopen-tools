// Function: sub_31FE5D0
// Address: 0x31fe5d0
//
void __fastcall sub_31FE5D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  char v14; // al
  int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  char v26; // al
  int v27; // ebx
  int v28; // eax
  unsigned __int64 v29[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v6 )
  {
    v19 = sub_C8D7D0(a1, a1 + 16, 0, 0x58u, v29, a6);
    v20 = *(unsigned int *)(a1 + 8);
    v21 = 5 * v20;
    v22 = v19 + 88 * v20;
    if ( v22 )
    {
      v23 = *a2;
      *(_QWORD *)(v22 + 24) = 0;
      *(_QWORD *)(v22 + 16) = 0;
      *(_DWORD *)(v22 + 32) = 0;
      *(_QWORD *)v22 = v23;
      *(_QWORD *)(v22 + 8) = 1;
      v24 = a2[2];
      ++a2[1];
      v25 = *(_QWORD *)(v22 + 16);
      *(_QWORD *)(v22 + 16) = v24;
      LODWORD(v24) = *((_DWORD *)a2 + 6);
      a2[2] = v25;
      LODWORD(v25) = *(_DWORD *)(v22 + 24);
      *(_DWORD *)(v22 + 24) = v24;
      LODWORD(v24) = *((_DWORD *)a2 + 7);
      *((_DWORD *)a2 + 6) = v25;
      LODWORD(v25) = *(_DWORD *)(v22 + 28);
      *(_DWORD *)(v22 + 28) = v24;
      LODWORD(v24) = *((_DWORD *)a2 + 8);
      *((_DWORD *)a2 + 7) = v25;
      LODWORD(v25) = *(_DWORD *)(v22 + 32);
      *(_DWORD *)(v22 + 32) = v24;
      *((_DWORD *)a2 + 8) = v25;
      *(_QWORD *)(v22 + 40) = v22 + 56;
      *(_QWORD *)(v22 + 48) = 0;
      v21 = *((unsigned int *)a2 + 12);
      if ( (_DWORD)v21 )
        sub_31FDD40(v22 + 40, (__int64)(a2 + 5), v21, v16, v17, v18);
      v26 = *((_BYTE *)a2 + 56);
      *(_BYTE *)(v22 + 80) = 0;
      *(_BYTE *)(v22 + 56) = v26;
      if ( *((_BYTE *)a2 + 80) )
      {
        v28 = *((_DWORD *)a2 + 18);
        *((_DWORD *)a2 + 18) = 0;
        *(_DWORD *)(v22 + 72) = v28;
        *(_QWORD *)(v22 + 64) = a2[8];
        LOBYTE(v28) = *((_BYTE *)a2 + 76);
        *(_BYTE *)(v22 + 80) = 1;
        *(_BYTE *)(v22 + 76) = v28;
      }
    }
    sub_31FE0C0((__int64 **)a1, v19, v21, v16, v17, v18);
    v27 = v29[0];
    if ( a1 + 16 != *(_QWORD *)a1 )
      _libc_free(*(_QWORD *)a1);
    *(_QWORD *)a1 = v19;
    *(_DWORD *)(a1 + 12) = v27;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v7 = *(_DWORD *)(a1 + 8);
    v8 = 11 * v6;
    v9 = *(_QWORD *)a1 + 88 * v6;
    if ( v9 )
    {
      v10 = *a2;
      *(_QWORD *)(v9 + 24) = 0;
      *(_QWORD *)(v9 + 16) = 0;
      *(_DWORD *)(v9 + 32) = 0;
      *(_QWORD *)v9 = v10;
      *(_QWORD *)(v9 + 8) = 1;
      v11 = a2[2];
      ++a2[1];
      v12 = *(_QWORD *)(v9 + 16);
      *(_QWORD *)(v9 + 16) = v11;
      LODWORD(v11) = *((_DWORD *)a2 + 6);
      a2[2] = v12;
      LODWORD(v12) = *(_DWORD *)(v9 + 24);
      *(_DWORD *)(v9 + 24) = v11;
      LODWORD(v11) = *((_DWORD *)a2 + 7);
      *((_DWORD *)a2 + 6) = v12;
      LODWORD(v12) = *(_DWORD *)(v9 + 28);
      *(_DWORD *)(v9 + 28) = v11;
      v13 = *((unsigned int *)a2 + 8);
      *((_DWORD *)a2 + 7) = v12;
      LODWORD(v12) = *(_DWORD *)(v9 + 32);
      *(_DWORD *)(v9 + 32) = v13;
      *((_DWORD *)a2 + 8) = v12;
      *(_QWORD *)(v9 + 40) = v9 + 56;
      *(_QWORD *)(v9 + 48) = 0;
      if ( *((_DWORD *)a2 + 12) )
        sub_31FDD40(v9 + 40, (__int64)(a2 + 5), v13, v8, a5, a6);
      v14 = *((_BYTE *)a2 + 56);
      *(_BYTE *)(v9 + 80) = 0;
      *(_BYTE *)(v9 + 56) = v14;
      if ( *((_BYTE *)a2 + 80) )
      {
        v15 = *((_DWORD *)a2 + 18);
        *((_DWORD *)a2 + 18) = 0;
        *(_DWORD *)(v9 + 72) = v15;
        *(_QWORD *)(v9 + 64) = a2[8];
        LOBYTE(v15) = *((_BYTE *)a2 + 76);
        *(_BYTE *)(v9 + 80) = 1;
        *(_BYTE *)(v9 + 76) = v15;
      }
      v7 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v7 + 1;
  }
}
