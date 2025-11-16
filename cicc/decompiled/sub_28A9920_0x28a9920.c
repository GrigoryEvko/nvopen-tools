// Function: sub_28A9920
// Address: 0x28a9920
//
void __fastcall sub_28A9920(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  __int16 v17; // dx
  __int64 v18; // rdx
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  int v21; // ebx
  unsigned __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0xB0u, v23, a6);
  v11 = *(unsigned int *)(a1 + 8);
  v12 = v8;
  v13 = *(_QWORD *)a1;
  v14 = 5 * v11;
  v15 = *(_QWORD *)a1 + 176 * v11;
  if ( *(_QWORD *)a1 != v15 )
  {
    v16 = v12;
    do
    {
      if ( v16 )
      {
        *(_QWORD *)v16 = *(_QWORD *)v13;
        *(_QWORD *)(v16 + 8) = *(_QWORD *)(v13 + 8);
        *(_QWORD *)(v16 + 16) = *(_QWORD *)(v13 + 16);
        v17 = *(_WORD *)(v13 + 24);
        *(_DWORD *)(v16 + 40) = 0;
        *(_WORD *)(v16 + 24) = v17;
        *(_QWORD *)(v16 + 32) = v16 + 48;
        *(_DWORD *)(v16 + 44) = 16;
        v18 = *(unsigned int *)(v13 + 40);
        if ( (_DWORD)v18 )
        {
          v22 = v13;
          sub_28A9600(v16 + 32, (char **)(v13 + 32), v18, v14, v9, v10);
          v13 = v22;
        }
      }
      v13 += 176LL;
      v16 += 176;
    }
    while ( v15 != v13 );
    v19 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a1 + 176LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v15 -= 176LL;
        v20 = *(_QWORD *)(v15 + 32);
        if ( v20 != v15 + 48 )
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
