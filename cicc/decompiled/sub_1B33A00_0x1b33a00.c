// Function: sub_1B33A00
// Address: 0x1b33a00
//
__int64 __fastcall sub_1B33A00(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rcx
  unsigned __int64 v8; // r14
  _QWORD *v9; // rax
  _QWORD *v10; // rcx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 *v13; // rax
  unsigned __int64 *v14; // r12
  __int64 v16; // [rsp+8h] [rbp-38h]

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v2 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v3 = ((v2
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v2
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = a2;
  v5 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v5 >= a2 )
    v4 = v5;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v16 = malloc(8 * v4);
  if ( !v16 )
    sub_16BD1C0("Allocation failed", 1u);
  v6 = *(_QWORD **)a1;
  v7 = 8LL * *(unsigned int *)(a1 + 8);
  v8 = *(_QWORD *)a1 + v7;
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = (_QWORD *)v16;
    v10 = (_QWORD *)(v16 + v7);
    do
    {
      if ( v9 )
      {
        *v9 = *v6;
        *v6 = 0;
      }
      ++v9;
      ++v6;
    }
    while ( v9 != v10 );
    v8 = *(_QWORD *)a1;
    v11 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( v11 != *(_QWORD *)a1 )
    {
      do
      {
        v12 = *(_QWORD *)(v11 - 8);
        v11 -= 8;
        if ( (v12 & 4) != 0 )
        {
          v13 = (unsigned __int64 *)(v12 & 0xFFFFFFFFFFFFFFF8LL);
          v14 = v13;
          if ( v13 )
          {
            if ( (unsigned __int64 *)*v13 != v13 + 2 )
              _libc_free(*v13);
            j_j___libc_free_0(v14, 48);
          }
        }
      }
      while ( v8 != v11 );
      v8 = *(_QWORD *)a1;
    }
  }
  if ( v8 != a1 + 16 )
    _libc_free(v8);
  *(_DWORD *)(a1 + 12) = v4;
  *(_QWORD *)a1 = v16;
  return v16;
}
