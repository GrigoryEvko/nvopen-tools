// Function: sub_1A01E90
// Address: 0x1a01e90
//
void __fastcall sub_1A01E90(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  int v10; // ecx
  unsigned __int64 v11; // r15
  __int64 v12; // rdi

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = (((((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
         | (*(unsigned int *)(a1 + 12) + 2LL)
         | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | (((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = a2;
  v5 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v5 >= a2 )
    v4 = v5;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v6 = malloc(40 * v4);
  if ( !v6 )
    sub_16BD1C0("Allocation failed", 1u);
  v7 = *(_QWORD *)a1;
  v8 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = v6;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = *(_QWORD *)v7;
        *(_QWORD *)(v9 + 8) = *(_QWORD *)(v7 + 8);
        *(_DWORD *)(v9 + 24) = *(_DWORD *)(v7 + 24);
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v7 + 16);
        v10 = *(_DWORD *)(v7 + 32);
        *(_DWORD *)(v7 + 24) = 0;
        *(_DWORD *)(v9 + 32) = v10;
        *(_BYTE *)(v9 + 36) = *(_BYTE *)(v7 + 36);
      }
      v7 += 40LL;
      v9 += 40;
    }
    while ( v8 != v7 );
    v11 = *(_QWORD *)a1;
    v8 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v8 -= 40LL;
        if ( *(_DWORD *)(v8 + 24) > 0x40u )
        {
          v12 = *(_QWORD *)(v8 + 16);
          if ( v12 )
            j_j___libc_free_0_0(v12);
        }
      }
      while ( v8 != v11 );
      v8 = *(_QWORD *)a1;
    }
  }
  if ( v8 != a1 + 16 )
    _libc_free(v8);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v4;
}
