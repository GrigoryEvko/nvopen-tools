// Function: sub_1EF9C40
// Address: 0x1ef9c40
//
void __fastcall sub_1EF9C40(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // r14
  __int64 v11; // rcx
  __int64 v12; // rdx
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rdi

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
        *(_DWORD *)v12 = *(_DWORD *)v8;
        *(_DWORD *)(v12 + 4) = *(_DWORD *)(v8 + 4);
        *(__m128i *)(v12 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
        *(_DWORD *)(v12 + 24) = *(_DWORD *)(v8 + 24);
        *(_QWORD *)(v8 + 8) = 0;
        *(_QWORD *)(v8 + 16) = 0;
        *(_DWORD *)(v8 + 24) = 0;
      }
      v12 += 32;
      v8 += 32LL;
    }
    while ( v12 != v11 );
    v13 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v14 = *(_QWORD *)(v10 - 24);
        v10 -= 32LL;
        _libc_free(v14);
      }
      while ( v10 != v13 );
      v10 = *(_QWORD *)a1;
    }
  }
  if ( v10 != a1 + 16 )
    _libc_free(v10);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v5;
}
