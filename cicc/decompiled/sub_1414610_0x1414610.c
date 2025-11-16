// Function: sub_1414610
// Address: 0x1414610
//
__int64 __fastcall sub_1414610(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // rax
  char **v12; // rsi
  __int64 v13; // rdi
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rdi
  __int64 v17; // [rsp+8h] [rbp-38h]

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
  v17 = malloc(88 * v7);
  if ( !v17 )
    sub_16BD1C0("Allocation failed");
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = v17;
    do
    {
      while ( 1 )
      {
        if ( v10 )
        {
          *(_QWORD *)v10 = *(_QWORD *)v8;
          *(_QWORD *)(v10 + 8) = *(_QWORD *)(v8 + 8);
          *(_QWORD *)(v10 + 16) = *(_QWORD *)(v8 + 16);
          *(_QWORD *)(v10 + 24) = *(_QWORD *)(v8 + 24);
          v11 = *(_QWORD *)(v8 + 32);
          *(_DWORD *)(v10 + 48) = 0;
          *(_QWORD *)(v10 + 32) = v11;
          *(_QWORD *)(v10 + 40) = v10 + 56;
          *(_DWORD *)(v10 + 52) = 4;
          if ( *(_DWORD *)(v8 + 48) )
            break;
        }
        v8 += 88LL;
        v10 += 88;
        if ( v9 == v8 )
          goto LABEL_15;
      }
      v12 = (char **)(v8 + 40);
      v13 = v10 + 40;
      v8 += 88LL;
      v10 += 88;
      sub_14117E0(v13, v12);
    }
    while ( v9 != v8 );
LABEL_15:
    v14 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v9 -= 88LL;
        v15 = *(_QWORD *)(v9 + 40);
        if ( v15 != v9 + 56 )
          _libc_free(v15);
      }
      while ( v9 != v14 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v7;
  *(_QWORD *)a1 = v17;
  return v17;
}
