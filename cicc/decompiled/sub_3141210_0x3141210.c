// Function: sub_3141210
// Address: 0x3141210
//
__int64 __fastcall sub_3141210(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // r15
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rsi
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rsi
  __int64 v19; // rdi
  int v20; // ebx
  __int64 v22; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v23 = a1 + 16;
  v7 = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v24, a6);
  v8 = *(_QWORD *)a1;
  v22 = v7;
  v9 = 32LL * *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + v9;
  if ( v8 != v8 + v9 )
  {
    v11 = v7 + v9;
    do
    {
      if ( v7 )
      {
        *(_DWORD *)(v7 + 24) = 0;
        *(_QWORD *)(v7 + 8) = 0;
        *(_DWORD *)(v7 + 16) = 0;
        *(_DWORD *)(v7 + 20) = 0;
        *(_QWORD *)v7 = 1;
        v12 = *(_QWORD *)(v8 + 8);
        ++*(_QWORD *)v8;
        v13 = *(_QWORD *)(v7 + 8);
        *(_QWORD *)(v7 + 8) = v12;
        LODWORD(v12) = *(_DWORD *)(v8 + 16);
        *(_QWORD *)(v8 + 8) = v13;
        LODWORD(v13) = *(_DWORD *)(v7 + 16);
        *(_DWORD *)(v7 + 16) = v12;
        LODWORD(v12) = *(_DWORD *)(v8 + 20);
        *(_DWORD *)(v8 + 16) = v13;
        LODWORD(v13) = *(_DWORD *)(v7 + 20);
        *(_DWORD *)(v7 + 20) = v12;
        LODWORD(v12) = *(_DWORD *)(v8 + 24);
        *(_DWORD *)(v8 + 20) = v13;
        LODWORD(v13) = *(_DWORD *)(v7 + 24);
        *(_DWORD *)(v7 + 24) = v12;
        *(_DWORD *)(v8 + 24) = v13;
      }
      v7 += 32;
      v8 += 32LL;
    }
    while ( v7 != v11 );
    v14 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v15 = *(unsigned int *)(v10 - 8);
        v10 -= 32LL;
        if ( (_DWORD)v15 )
        {
          v16 = *(_QWORD *)(v10 + 8);
          v17 = v16 + 48 * v15;
          do
          {
            while ( *(_QWORD *)v16 == -1 || *(_QWORD *)v16 == -2 )
            {
              v16 += 48;
              if ( v17 == v16 )
                goto LABEL_13;
            }
            v18 = *(unsigned int *)(v16 + 40);
            v19 = *(_QWORD *)(v16 + 24);
            v16 += 48;
            sub_C7D6A0(v19, 32 * v18, 8);
          }
          while ( v17 != v16 );
        }
LABEL_13:
        sub_C7D6A0(*(_QWORD *)(v10 + 8), 48LL * *(unsigned int *)(v10 + 24), 8);
      }
      while ( v10 != v14 );
      v10 = *(_QWORD *)a1;
    }
  }
  v20 = v24[0];
  if ( v23 != v10 )
    _libc_free(v10);
  *(_DWORD *)(a1 + 12) = v20;
  *(_QWORD *)a1 = v22;
  return v22;
}
