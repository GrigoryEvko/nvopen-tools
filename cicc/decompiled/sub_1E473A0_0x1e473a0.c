// Function: sub_1E473A0
// Address: 0x1e473a0
//
__int64 __fastcall sub_1E473A0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // r12
  _QWORD *v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  __int64 v14; // [rsp+8h] [rbp-38h]

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
  v14 = malloc(72 * v7);
  if ( !v14 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = (_QWORD *)v14;
    v10 = *(_QWORD *)a1;
    do
    {
      if ( v9 )
        sub_16CCEE0(v9, (__int64)(v9 + 5), 4, v10);
      v10 += 72;
      v9 += 9;
    }
    while ( v8 != v10 );
    v11 = *(_QWORD *)a1;
    v8 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v8 -= 72LL;
        v12 = *(_QWORD *)(v8 + 16);
        if ( v12 != *(_QWORD *)(v8 + 8) )
          _libc_free(v12);
      }
      while ( v8 != v11 );
      v8 = *(_QWORD *)a1;
    }
  }
  if ( v8 != a1 + 16 )
    _libc_free(v8);
  *(_DWORD *)(a1 + 12) = v7;
  *(_QWORD *)a1 = v14;
  return v14;
}
