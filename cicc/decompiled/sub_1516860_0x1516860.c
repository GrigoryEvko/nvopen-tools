// Function: sub_1516860
// Address: 0x1516860
//
__int64 __fastcall sub_1516860(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  __int64 *v8; // rbx
  __int64 *v9; // r12
  _QWORD *v10; // r15
  __int64 v11; // rsi
  __int64 *v12; // rbx
  __int64 v13; // rdi
  __int64 v15; // [rsp+8h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
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
  v15 = malloc(16 * v7);
  if ( !v15 )
    sub_16BD1C0("Allocation failed");
  v8 = *(__int64 **)a1;
  v9 = (__int64 *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
  if ( *(__int64 **)a1 != v9 )
  {
    v10 = (_QWORD *)v15;
    do
    {
      if ( v10 )
      {
        v11 = *v8;
        *v10 = *v8;
        if ( v11 )
        {
          sub_1623210(v8, v11, v10);
          *v8 = 0;
        }
        v10[1] = v8[1];
        v8[1] = 0;
      }
      v8 += 2;
      v10 += 2;
    }
    while ( v9 != v8 );
    v12 = *(__int64 **)a1;
    v9 = (__int64 *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
    if ( *(__int64 **)a1 != v9 )
    {
      do
      {
        v13 = *(v9 - 1);
        v9 -= 2;
        if ( v13 )
          sub_16307F0();
        if ( *v9 )
          sub_161E7C0(v9);
      }
      while ( v9 != v12 );
      v9 = *(__int64 **)a1;
    }
  }
  if ( v9 != (__int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v9);
  *(_DWORD *)(a1 + 12) = v7;
  *(_QWORD *)a1 = v15;
  return v15;
}
