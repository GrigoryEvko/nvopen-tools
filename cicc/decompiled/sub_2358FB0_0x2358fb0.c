// Function: sub_2358FB0
// Address: 0x2358fb0
//
void __fastcall sub_2358FB0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // r13
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r12
  __int64 v11; // rdx
  int v12; // ecx
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  int v16; // r15d
  unsigned __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v17, a6);
  v9 = *(_QWORD *)a1;
  v10 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = v8;
    do
    {
      if ( v11 )
      {
        *(_QWORD *)v11 = *(_QWORD *)v9;
        *(_DWORD *)(v11 + 16) = *(_DWORD *)(v9 + 16);
        *(_QWORD *)(v11 + 8) = *(_QWORD *)(v9 + 8);
        v12 = *(_DWORD *)(v9 + 32);
        *(_DWORD *)(v9 + 16) = 0;
        *(_DWORD *)(v11 + 32) = v12;
        *(_QWORD *)(v11 + 24) = *(_QWORD *)(v9 + 24);
        *(_DWORD *)(v9 + 32) = 0;
      }
      v9 += 40LL;
      v11 += 40;
    }
    while ( v10 != v9 );
    v13 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v10 -= 40LL;
        if ( *(_DWORD *)(v10 + 32) > 0x40u )
        {
          v14 = *(_QWORD *)(v10 + 24);
          if ( v14 )
            j_j___libc_free_0_0(v14);
        }
        if ( *(_DWORD *)(v10 + 16) > 0x40u )
        {
          v15 = *(_QWORD *)(v10 + 8);
          if ( v15 )
            j_j___libc_free_0_0(v15);
        }
      }
      while ( v10 != v13 );
      v10 = *(_QWORD *)a1;
    }
  }
  v16 = v17[0];
  if ( v6 != v10 )
    _libc_free(v10);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v16;
}
