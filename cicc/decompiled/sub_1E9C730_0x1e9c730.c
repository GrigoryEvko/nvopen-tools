// Function: sub_1E9C730
// Address: 0x1e9c730
//
void __fastcall sub_1E9C730(__int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned __int64 i; // rax
  char v9; // cl

  v2 = (((((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
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
  v3 = (v2 | (v2 >> 16) | HIDWORD(v2)) + 1;
  if ( v3 > 0xFFFFFFFF )
    v3 = 0xFFFFFFFFLL;
  v4 = malloc(24 * v3);
  if ( !v4 )
    sub_16BD1C0("Allocation failed", 1u);
  v5 = *(_QWORD *)a1;
  v6 = v4;
  v7 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  for ( i = *(_QWORD *)a1; v7 != i; v6 += 24 )
  {
    if ( v6 )
    {
      *(_QWORD *)v6 = *(_QWORD *)i;
      v9 = *(_BYTE *)(i + 16);
      *(_BYTE *)(v6 + 16) = v9;
      if ( v9 )
        *(_QWORD *)(v6 + 8) = *(_QWORD *)(i + 8);
    }
    i += 24LL;
  }
  if ( v5 != a1 + 16 )
    _libc_free(v5);
  *(_QWORD *)a1 = v4;
  *(_DWORD *)(a1 + 12) = v3;
}
