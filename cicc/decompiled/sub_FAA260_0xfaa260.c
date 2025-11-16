// Function: sub_FAA260
// Address: 0xfaa260
//
__int64 __fastcall sub_FAA260(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  char **v8; // rsi
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // rdi
  int v24; // ebx
  __int64 v26; // [rsp+8h] [rbp-48h]
  unsigned __int64 v27[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = (char **)(a1 + 16);
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x48u, v27, a6);
  v14 = *(_QWORD *)a1;
  v26 = v10;
  v15 = v10;
  v16 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v16 )
  {
    do
    {
      while ( 1 )
      {
        if ( v15 )
        {
          v17 = *(_QWORD *)v14;
          *(_DWORD *)(v15 + 32) = 0;
          *(_QWORD *)(v15 + 16) = 0;
          *(_DWORD *)(v15 + 24) = 0;
          *(_DWORD *)(v15 + 28) = 0;
          *(_QWORD *)v15 = v17;
          *(_QWORD *)(v15 + 8) = 1;
          v18 = *(_QWORD *)(v14 + 16);
          ++*(_QWORD *)(v14 + 8);
          v19 = *(_QWORD *)(v15 + 16);
          *(_QWORD *)(v15 + 16) = v18;
          LODWORD(v18) = *(_DWORD *)(v14 + 24);
          *(_QWORD *)(v14 + 16) = v19;
          LODWORD(v19) = *(_DWORD *)(v15 + 24);
          *(_DWORD *)(v15 + 24) = v18;
          LODWORD(v18) = *(_DWORD *)(v14 + 28);
          *(_DWORD *)(v14 + 24) = v19;
          LODWORD(v19) = *(_DWORD *)(v15 + 28);
          *(_DWORD *)(v15 + 28) = v18;
          v20 = *(unsigned int *)(v14 + 32);
          *(_DWORD *)(v14 + 28) = v19;
          LODWORD(v19) = *(_DWORD *)(v15 + 32);
          *(_DWORD *)(v15 + 32) = v20;
          *(_DWORD *)(v14 + 32) = v19;
          *(_QWORD *)(v15 + 40) = v15 + 56;
          *(_DWORD *)(v15 + 48) = 0;
          *(_DWORD *)(v15 + 52) = 2;
          if ( *(_DWORD *)(v14 + 48) )
            break;
        }
        v14 += 72;
        v15 += 72;
        if ( v16 == v14 )
          goto LABEL_7;
      }
      v8 = (char **)(v14 + 40);
      v21 = v15 + 40;
      v14 += 72;
      v15 += 72;
      sub_F8F130(v21, v8, v20, v11, v12, v13);
    }
    while ( v16 != v14 );
LABEL_7:
    v22 = *(_QWORD *)a1;
    v16 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v16 )
    {
      do
      {
        v16 -= 72;
        v23 = *(_QWORD *)(v16 + 40);
        if ( v23 != v16 + 56 )
          _libc_free(v23, v8);
        v8 = (char **)(8LL * *(unsigned int *)(v16 + 32));
        sub_C7D6A0(*(_QWORD *)(v16 + 16), (__int64)v8, 8);
      }
      while ( v16 != v22 );
      v16 = *(_QWORD *)a1;
    }
  }
  v24 = v27[0];
  if ( v7 != v16 )
    _libc_free(v16, v8);
  *(_DWORD *)(a1 + 12) = v24;
  *(_QWORD *)a1 = v26;
  return v26;
}
