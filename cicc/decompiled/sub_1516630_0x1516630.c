// Function: sub_1516630
// Address: 0x1516630
//
__int64 __fastcall sub_1516630(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  __int64 *v7; // rbx
  unsigned __int64 v8; // r12
  _QWORD *v9; // r15
  __int64 v10; // rsi
  __int64 *v11; // rbx
  __int64 v12; // rsi
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
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
  v5 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v5 >= a2 )
    v2 = v5;
  v6 = v2;
  if ( v2 > 0xFFFFFFFF )
    v6 = 0xFFFFFFFFLL;
  v14 = malloc(8 * v6);
  if ( !v14 )
    sub_16BD1C0("Allocation failed");
  v7 = *(__int64 **)a1;
  v8 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = (_QWORD *)v14;
    do
    {
      if ( v9 )
      {
        v10 = *v7;
        *v9 = *v7;
        if ( v10 )
        {
          sub_1623210(v7, v10, v9);
          *v7 = 0;
        }
      }
      ++v7;
      ++v9;
    }
    while ( (__int64 *)v8 != v7 );
    v11 = *(__int64 **)a1;
    v8 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v12 = *(_QWORD *)(v8 - 8);
        v8 -= 8LL;
        if ( v12 )
          sub_161E7C0(v8);
      }
      while ( (__int64 *)v8 != v11 );
      v8 = *(_QWORD *)a1;
    }
  }
  if ( v8 != a1 + 16 )
    _libc_free(v8);
  *(_DWORD *)(a1 + 12) = v6;
  *(_QWORD *)a1 = v14;
  return v14;
}
