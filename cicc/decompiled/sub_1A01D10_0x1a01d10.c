// Function: sub_1A01D10
// Address: 0x1a01d10
//
void __fastcall sub_1A01D10(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  unsigned __int64 v10; // r15
  __int64 v11; // rdi

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
  v6 = malloc(24 * v4);
  if ( !v6 )
    sub_16BD1C0("Allocation failed", 1u);
  v7 = *(_QWORD *)a1;
  v8 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = v6;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = *(_QWORD *)v7;
        *(_DWORD *)(v9 + 16) = *(_DWORD *)(v7 + 16);
        *(_QWORD *)(v9 + 8) = *(_QWORD *)(v7 + 8);
        *(_DWORD *)(v7 + 16) = 0;
      }
      v7 += 24LL;
      v9 += 24;
    }
    while ( v8 != v7 );
    v10 = *(_QWORD *)a1;
    v8 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v8 -= 24LL;
        if ( *(_DWORD *)(v8 + 16) > 0x40u )
        {
          v11 = *(_QWORD *)(v8 + 8);
          if ( v11 )
            j_j___libc_free_0_0(v11);
        }
      }
      while ( v8 != v10 );
      v8 = *(_QWORD *)a1;
    }
  }
  if ( v8 != a1 + 16 )
    _libc_free(v8);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v4;
}
