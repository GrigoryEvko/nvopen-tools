// Function: sub_27A3A70
// Address: 0x27a3a70
//
void __fastcall sub_27A3A70(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r14
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdi
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  int v21; // ebx
  unsigned __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v23, a6);
  v11 = *(unsigned int *)(a1 + 8);
  v12 = v8;
  v13 = *(_QWORD *)a1;
  v14 = *(_QWORD *)a1 + 56 * v11;
  if ( *(_QWORD *)a1 != v14 )
  {
    v15 = v12;
    do
    {
      while ( 1 )
      {
        if ( v15 )
        {
          v16 = *(_QWORD *)v13;
          *(_DWORD *)(v15 + 16) = 0;
          *(_DWORD *)(v15 + 20) = 4;
          *(_QWORD *)v15 = v16;
          *(_QWORD *)(v15 + 8) = v15 + 24;
          v17 = *(unsigned int *)(v13 + 16);
          if ( (_DWORD)v17 )
            break;
        }
        v13 += 56LL;
        v15 += 56;
        if ( v14 == v13 )
          goto LABEL_7;
      }
      v18 = v15 + 8;
      v22 = v13;
      v15 += 56;
      sub_27A0EB0(v18, (char **)(v13 + 8), v17, v11, v9, v10);
      v13 = v22 + 56;
    }
    while ( v14 != v22 + 56 );
LABEL_7:
    v19 = *(_QWORD *)a1;
    v14 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v14 )
    {
      do
      {
        v14 -= 56LL;
        v20 = *(_QWORD *)(v14 + 8);
        if ( v20 != v14 + 24 )
          _libc_free(v20);
      }
      while ( v14 != v19 );
      v14 = *(_QWORD *)a1;
    }
  }
  v21 = v23[0];
  if ( v6 != v14 )
    _libc_free(v14);
  *(_QWORD *)a1 = v12;
  *(_DWORD *)(a1 + 12) = v21;
}
