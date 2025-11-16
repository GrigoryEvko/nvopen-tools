// Function: sub_2EB5A00
// Address: 0x2eb5a00
//
__int64 __fastcall sub_2EB5A00(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  unsigned __int64 v13; // r12
  __int64 v14; // rbx
  unsigned __int64 v15; // r14
  __int64 v16; // rax
  char **v17; // rsi
  __int64 v18; // rdi
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  int v21; // ebx
  __int64 v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v24, a6);
  v12 = *(unsigned int *)(a1 + 8);
  v13 = *(_QWORD *)a1;
  v23 = v8;
  v14 = v8;
  v15 = *(_QWORD *)a1 + 56 * v12;
  if ( *(_QWORD *)a1 != v15 )
  {
    do
    {
      while ( 1 )
      {
        if ( v14 )
        {
          *(_DWORD *)v14 = *(_DWORD *)v13;
          *(_DWORD *)(v14 + 4) = *(_DWORD *)(v13 + 4);
          *(_DWORD *)(v14 + 8) = *(_DWORD *)(v13 + 8);
          *(_DWORD *)(v14 + 12) = *(_DWORD *)(v13 + 12);
          v16 = *(_QWORD *)(v13 + 16);
          *(_DWORD *)(v14 + 32) = 0;
          *(_QWORD *)(v14 + 16) = v16;
          *(_QWORD *)(v14 + 24) = v14 + 40;
          *(_DWORD *)(v14 + 36) = 4;
          if ( *(_DWORD *)(v13 + 32) )
            break;
        }
        v13 += 56LL;
        v14 += 56;
        if ( v15 == v13 )
          goto LABEL_7;
      }
      v17 = (char **)(v13 + 24);
      v18 = v14 + 24;
      v13 += 56LL;
      v14 += 56;
      sub_2EB32F0(v18, v17, v12, v9, v10, v11);
    }
    while ( v15 != v13 );
LABEL_7:
    v19 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v15 -= 56LL;
        v20 = *(_QWORD *)(v15 + 24);
        if ( v20 != v15 + 40 )
          _libc_free(v20);
      }
      while ( v15 != v19 );
      v15 = *(_QWORD *)a1;
    }
  }
  v21 = v24[0];
  if ( v6 != v15 )
    _libc_free(v15);
  *(_DWORD *)(a1 + 12) = v21;
  *(_QWORD *)a1 = v23;
  return v23;
}
