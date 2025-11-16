// Function: sub_1D4FA80
// Address: 0x1d4fa80
//
void __fastcall sub_1D4FA80(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // r14
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // rdx
  int v11; // ecx
  unsigned __int64 v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rdi

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
  v7 = malloc(40 * v5);
  if ( !v7 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = v7;
    do
    {
      if ( v10 )
      {
        *(_DWORD *)v10 = *(_DWORD *)v8;
        *(_DWORD *)(v10 + 16) = *(_DWORD *)(v8 + 16);
        *(_QWORD *)(v10 + 8) = *(_QWORD *)(v8 + 8);
        v11 = *(_DWORD *)(v8 + 32);
        *(_DWORD *)(v8 + 16) = 0;
        *(_DWORD *)(v10 + 32) = v11;
        *(_QWORD *)(v10 + 24) = *(_QWORD *)(v8 + 24);
        *(_DWORD *)(v8 + 32) = 0;
      }
      v8 += 40LL;
      v10 += 40;
    }
    while ( v9 != v8 );
    v12 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v9 -= 40LL;
        if ( *(_DWORD *)(v9 + 32) > 0x40u )
        {
          v13 = *(_QWORD *)(v9 + 24);
          if ( v13 )
            j_j___libc_free_0_0(v13);
        }
        if ( *(_DWORD *)(v9 + 16) > 0x40u )
        {
          v14 = *(_QWORD *)(v9 + 8);
          if ( v14 )
            j_j___libc_free_0_0(v14);
        }
      }
      while ( v9 != v12 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v5;
}
