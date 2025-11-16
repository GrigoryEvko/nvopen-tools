// Function: sub_2F8CBB0
// Address: 0x2f8cbb0
//
void __fastcall sub_2F8CBB0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  char v17; // dl
  __int64 v18; // rdx
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  int v21; // ebx
  unsigned __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x58u, v23, a6);
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
      if ( v16 )
      {
        *(_QWORD *)v16 = *(_QWORD *)v13;
        *(_DWORD *)(v16 + 8) = *(_DWORD *)(v13 + 8);
        v17 = *(_BYTE *)(v13 + 12);
        *(_DWORD *)(v16 + 24) = 0;
        *(_BYTE *)(v16 + 12) = v17;
        *(_QWORD *)(v16 + 16) = v16 + 32;
        *(_DWORD *)(v16 + 28) = 6;
        v18 = *(unsigned int *)(v13 + 24);
        if ( (_DWORD)v18 )
        {
          v22 = v13;
          sub_2F8ABB0(v16 + 16, (char **)(v13 + 16), v18, v14, v9, v10);
          v13 = v22;
        }
        *(_DWORD *)(v16 + 80) = *(_DWORD *)(v13 + 80);
      }
      v13 += 88LL;
      v16 += 88;
    }
    while ( v15 != v13 );
    v19 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v15 -= 88LL;
        v20 = *(_QWORD *)(v15 + 16);
        if ( v20 != v15 + 32 )
          _libc_free(v20);
      }
      while ( v15 != v19 );
      v15 = *(_QWORD *)a1;
    }
  }
  v21 = v23[0];
  if ( v6 != v15 )
    _libc_free(v15);
  *(_QWORD *)a1 = v12;
  *(_DWORD *)(a1 + 12) = v21;
}
