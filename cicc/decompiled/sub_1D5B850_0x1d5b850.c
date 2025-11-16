// Function: sub_1D5B850
// Address: 0x1d5b850
//
void __fastcall sub_1D5B850(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  __int64 v6; // r14
  __int64 *v7; // rdx
  __int64 v8; // rcx
  unsigned __int64 v9; // r12
  _QWORD *v10; // rcx
  _QWORD *v11; // rax
  __int64 *v12; // r15
  __int64 v13; // rdi

  v3 = ((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2;
  v4 = ((((v3 | (*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
       | v3
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | ((v3 | (*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
     | v3
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v5 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v5 > 0xFFFFFFFF )
    v5 = 0xFFFFFFFFLL;
  v6 = malloc(8 * v5);
  if ( !v6 )
  {
    a2 = 1;
    sub_16BD1C0("Allocation failed", 1u);
  }
  v7 = *(__int64 **)a1;
  v8 = 8LL * *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1 + v8;
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = (_QWORD *)(v6 + v8);
    v11 = (_QWORD *)v6;
    do
    {
      if ( v11 )
      {
        a2 = *v7;
        *v11 = *v7;
        *v7 = 0;
      }
      ++v11;
      ++v7;
    }
    while ( v11 != v10 );
    v12 = *(__int64 **)a1;
    v9 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v13 = *(_QWORD *)(v9 - 8);
        v9 -= 8LL;
        if ( v13 )
          (*(void (__fastcall **)(__int64, __int64, __int64 *))(*(_QWORD *)v13 + 8LL))(v13, a2, v7);
      }
      while ( v12 != (__int64 *)v9 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v5;
}
