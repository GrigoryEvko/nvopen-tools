// Function: sub_324BBF0
// Address: 0x324bbf0
//
void __fastcall sub_324BBF0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // r14
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  unsigned __int64 v15; // r12
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdi
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  int v22; // ebx
  unsigned __int64 v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x58u, v24, a6);
  v11 = *(unsigned int *)(a1 + 8);
  v12 = v8;
  v13 = *(_QWORD *)a1;
  v14 = 5 * v11;
  v15 = *(_QWORD *)a1 + 88 * v11;
  if ( *(_QWORD *)a1 != v15 )
  {
    v16 = v12;
    do
    {
      while ( 1 )
      {
        if ( v16 )
        {
          v17 = *(_QWORD *)v13;
          *(_DWORD *)(v16 + 16) = 0;
          *(_DWORD *)(v16 + 20) = 8;
          *(_QWORD *)v16 = v17;
          *(_QWORD *)(v16 + 8) = v16 + 24;
          v18 = *(unsigned int *)(v13 + 16);
          if ( (_DWORD)v18 )
            break;
        }
        v13 += 88LL;
        v16 += 88;
        if ( v15 == v13 )
          goto LABEL_7;
      }
      v19 = v16 + 8;
      v23 = v13;
      v16 += 88;
      sub_32473B0(v19, (char **)(v13 + 8), v18, v14, v9, v10);
      v13 = v23 + 88;
    }
    while ( v15 != v23 + 88 );
LABEL_7:
    v20 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v15 -= 88LL;
        v21 = *(_QWORD *)(v15 + 8);
        if ( v21 != v15 + 24 )
          _libc_free(v21);
      }
      while ( v15 != v20 );
      v15 = *(_QWORD *)a1;
    }
  }
  v22 = v24[0];
  if ( v6 != v15 )
    _libc_free(v15);
  *(_QWORD *)a1 = v12;
  *(_DWORD *)(a1 + 12) = v22;
}
