// Function: sub_102D890
// Address: 0x102d890
//
__int64 __fastcall sub_102D890(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  char **v8; // rsi
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rbx
  __int64 v22; // rdi
  int v23; // ebx
  __int64 v25; // [rsp+8h] [rbp-48h]
  unsigned __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = (char **)(a1 + 16);
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x58u, v26, a6);
  v14 = *(_QWORD *)a1;
  v25 = v10;
  v15 = v10;
  v16 = *(unsigned int *)(a1 + 8);
  v17 = 5 * v16;
  v18 = *(_QWORD *)a1 + 88 * v16;
  if ( *(_QWORD *)a1 != v18 )
  {
    do
    {
      while ( 1 )
      {
        if ( v15 )
        {
          *(_QWORD *)v15 = *(_QWORD *)v14;
          *(_QWORD *)(v15 + 8) = *(_QWORD *)(v14 + 8);
          *(_QWORD *)(v15 + 16) = *(_QWORD *)(v14 + 16);
          *(_QWORD *)(v15 + 24) = *(_QWORD *)(v14 + 24);
          v19 = *(_QWORD *)(v14 + 32);
          *(_DWORD *)(v15 + 48) = 0;
          *(_QWORD *)(v15 + 32) = v19;
          *(_QWORD *)(v15 + 40) = v15 + 56;
          *(_DWORD *)(v15 + 52) = 4;
          if ( *(_DWORD *)(v14 + 48) )
            break;
        }
        v14 += 88;
        v15 += 88;
        if ( v18 == v14 )
          goto LABEL_7;
      }
      v8 = (char **)(v14 + 40);
      v20 = v15 + 40;
      v14 += 88;
      v15 += 88;
      sub_1029680(v20, v8, v17, v11, v12, v13);
    }
    while ( v18 != v14 );
LABEL_7:
    v21 = *(_QWORD *)a1;
    v18 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v18 )
    {
      do
      {
        v18 -= 88;
        v22 = *(_QWORD *)(v18 + 40);
        if ( v22 != v18 + 56 )
          _libc_free(v22, v8);
      }
      while ( v18 != v21 );
      v18 = *(_QWORD *)a1;
    }
  }
  v23 = v26[0];
  if ( v7 != v18 )
    _libc_free(v18, v8);
  *(_DWORD *)(a1 + 12) = v23;
  *(_QWORD *)a1 = v25;
  return v25;
}
