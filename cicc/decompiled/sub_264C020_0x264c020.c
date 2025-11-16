// Function: sub_264C020
// Address: 0x264c020
//
void __fastcall sub_264C020(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r14
  __int64 v11; // rdx
  int v12; // ecx
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rdi
  int v15; // r15d
  unsigned __int64 v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v16, a6);
  v9 = *(_QWORD *)a1;
  v10 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = v8;
    do
    {
      if ( v11 )
      {
        *(_QWORD *)v11 = *(_QWORD *)v9;
        *(_QWORD *)(v11 + 8) = *(_QWORD *)(v9 + 8);
        *(_QWORD *)(v11 + 16) = *(_QWORD *)(v9 + 16);
        *(_QWORD *)(v11 + 24) = *(_QWORD *)(v9 + 24);
        v12 = *(_DWORD *)(v9 + 32);
        *(_QWORD *)(v9 + 24) = 0;
        *(_QWORD *)(v9 + 16) = 0;
        *(_QWORD *)(v9 + 8) = 0;
        *(_DWORD *)(v11 + 32) = v12;
        *(_QWORD *)(v11 + 40) = *(_QWORD *)(v9 + 40);
        *(_QWORD *)(v11 + 48) = *(_QWORD *)(v9 + 48);
      }
      v9 += 56LL;
      v11 += 56;
    }
    while ( v10 != v9 );
    v13 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v14 = *(_QWORD *)(v10 - 48);
        v10 -= 56LL;
        if ( v14 )
          j_j___libc_free_0(v14);
      }
      while ( v13 != v10 );
      v10 = *(_QWORD *)a1;
    }
  }
  v15 = v16[0];
  if ( v6 != v10 )
    _libc_free(v10);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v15;
}
