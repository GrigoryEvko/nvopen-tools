// Function: sub_1905050
// Address: 0x1905050
//
void __fastcall sub_1905050(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // r14
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // r12
  __int64 v11; // rcx
  __int64 v12; // rax
  int v13; // esi
  unsigned __int64 v14; // r15
  __int64 v15; // rdi

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v4 = ((v3
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v3
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v5 = a2;
  v6 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v6 >= a2 )
    v5 = v6;
  if ( v5 > 0xFFFFFFFF )
    v5 = 0xFFFFFFFFLL;
  v7 = malloc(32 * v5);
  if ( !v7 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *(_QWORD *)a1;
  v9 = 32LL * *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + v9;
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = v7 + v9;
    v12 = v7;
    do
    {
      if ( v12 )
      {
        *(_DWORD *)(v12 + 8) = *(_DWORD *)(v8 + 8);
        *(_QWORD *)v12 = *(_QWORD *)v8;
        v13 = *(_DWORD *)(v8 + 24);
        *(_DWORD *)(v8 + 8) = 0;
        *(_DWORD *)(v12 + 24) = v13;
        *(_QWORD *)(v12 + 16) = *(_QWORD *)(v8 + 16);
        *(_DWORD *)(v8 + 24) = 0;
      }
      v12 += 32;
      v8 += 32LL;
    }
    while ( v12 != v11 );
    v14 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v10 -= 32LL;
        if ( *(_DWORD *)(v10 + 24) > 0x40u )
        {
          v15 = *(_QWORD *)(v10 + 16);
          if ( v15 )
            j_j___libc_free_0_0(v15);
        }
        if ( *(_DWORD *)(v10 + 8) > 0x40u && *(_QWORD *)v10 )
          j_j___libc_free_0_0(*(_QWORD *)v10);
      }
      while ( v10 != v14 );
      v10 = *(_QWORD *)a1;
    }
  }
  if ( v10 != a1 + 16 )
    _libc_free(v10);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v5;
}
