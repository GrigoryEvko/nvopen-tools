// Function: sub_2758F00
// Address: 0x2758f00
//
__int64 __fastcall sub_2758F00(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // r13
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // rdi
  int v24; // r12d
  __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  unsigned __int64 v28; // [rsp+18h] [rbp-48h]
  unsigned __int64 v29[7]; // [rsp+28h] [rbp-38h] BYREF

  v27 = a1 + 16;
  v7 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v29, a6);
  v10 = *(_QWORD *)a1;
  v26 = v7;
  v11 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v11 )
  {
    do
    {
      while ( 1 )
      {
        v12 = 56;
        if ( v7 )
        {
          v13 = *(_QWORD *)v10;
          *(_DWORD *)(v7 + 32) = 0;
          v12 = v7 + 56;
          *(_QWORD *)(v7 + 16) = 0;
          *(_DWORD *)(v7 + 24) = 0;
          *(_DWORD *)(v7 + 28) = 0;
          *(_QWORD *)v7 = v13;
          *(_QWORD *)(v7 + 8) = 1;
          v14 = *(_QWORD *)(v10 + 16);
          ++*(_QWORD *)(v10 + 8);
          v15 = *(_QWORD *)(v7 + 16);
          *(_QWORD *)(v7 + 16) = v14;
          LODWORD(v14) = *(_DWORD *)(v10 + 24);
          *(_QWORD *)(v10 + 16) = v15;
          LODWORD(v15) = *(_DWORD *)(v7 + 24);
          *(_DWORD *)(v7 + 24) = v14;
          LODWORD(v14) = *(_DWORD *)(v10 + 28);
          *(_DWORD *)(v10 + 24) = v15;
          LODWORD(v15) = *(_DWORD *)(v7 + 28);
          *(_DWORD *)(v7 + 28) = v14;
          v16 = *(unsigned int *)(v10 + 32);
          *(_DWORD *)(v10 + 28) = v15;
          LODWORD(v15) = *(_DWORD *)(v7 + 32);
          *(_DWORD *)(v7 + 32) = v16;
          *(_DWORD *)(v10 + 32) = v15;
          *(_QWORD *)(v7 + 40) = v7 + 56;
          *(_DWORD *)(v7 + 48) = 0;
          *(_DWORD *)(v7 + 52) = 0;
          v17 = *(unsigned int *)(v10 + 48);
          if ( (_DWORD)v17 )
            break;
        }
        v10 += 56LL;
        v7 = v12;
        if ( v11 == v10 )
          goto LABEL_7;
      }
      v18 = v10 + 40;
      v10 += 56LL;
      sub_27589D0(v7 + 40, v18, v17, v16, v8, v9);
      v7 = v12;
    }
    while ( v11 != v10 );
LABEL_7:
    v28 = *(_QWORD *)a1;
    v11 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v11 )
    {
      do
      {
        v19 = *(unsigned int *)(v11 - 8);
        v20 = *(_QWORD *)(v11 - 16);
        v11 -= 56LL;
        v21 = v20 + 56 * v19;
        if ( v20 != v21 )
        {
          do
          {
            v22 = *(_QWORD *)(v21 - 32);
            v21 -= 56LL;
            while ( v22 )
            {
              sub_2754510(*(_QWORD *)(v22 + 24));
              v23 = v22;
              v22 = *(_QWORD *)(v22 + 16);
              j_j___libc_free_0(v23);
            }
          }
          while ( v20 != v21 );
          v20 = *(_QWORD *)(v11 + 40);
        }
        if ( v20 != v11 + 56 )
          _libc_free(v20);
        sub_C7D6A0(*(_QWORD *)(v11 + 16), 16LL * *(unsigned int *)(v11 + 32), 8);
      }
      while ( v11 != v28 );
      v11 = *(_QWORD *)a1;
    }
  }
  v24 = v29[0];
  if ( v27 != v11 )
    _libc_free(v11);
  *(_DWORD *)(a1 + 12) = v24;
  *(_QWORD *)a1 = v26;
  return v26;
}
