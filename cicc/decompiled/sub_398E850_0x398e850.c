// Function: sub_398E850
// Address: 0x398e850
//
__int64 __fastcall sub_398E850(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r12
  _QWORD *v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // r14
  _QWORD *v11; // rax
  _QWORD *v12; // rcx
  __int64 v13; // rbx
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // rdi
  __int64 v17; // [rsp+8h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v4 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v5 = ((v4
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v4
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v6 = (v5 | (v5 >> 16) | HIDWORD(v5)) + 1;
  if ( v6 >= a2 )
    v3 = v6;
  v7 = v3;
  if ( v3 > 0xFFFFFFFF )
    v7 = 0xFFFFFFFFLL;
  v17 = malloc(8 * v7);
  if ( !v17 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *(_QWORD **)a1;
  v9 = 8LL * *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + v9;
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = (_QWORD *)v17;
    v12 = (_QWORD *)(v17 + v9);
    do
    {
      if ( v11 )
      {
        *v11 = *v8;
        *v8 = 0;
      }
      ++v11;
      ++v8;
    }
    while ( v11 != v12 );
    v10 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( v13 != *(_QWORD *)a1 )
    {
      do
      {
        v14 = *(_QWORD *)(v13 - 8);
        v13 -= 8;
        if ( v14 )
        {
          v15 = *(_QWORD *)(v14 + 40);
          if ( v15 != v14 + 56 )
            _libc_free(v15);
          j_j___libc_free_0(v14);
        }
      }
      while ( v10 != v13 );
      v10 = *(_QWORD *)a1;
    }
  }
  if ( v10 != a1 + 16 )
    _libc_free(v10);
  *(_DWORD *)(a1 + 12) = v7;
  *(_QWORD *)a1 = v17;
  return v17;
}
