// Function: sub_F91440
// Address: 0xf91440
//
bool __fastcall sub_F91440(__int64 **a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // r14
  unsigned __int64 v5; // rax
  int v6; // edx
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdi
  _QWORD *v14; // rdx
  __int64 v15; // rsi
  _QWORD *v16; // rax
  __int64 v17; // rax
  __int64 i; // rdx
  __int64 v19; // rax
  __int64 j; // rsi
  __int64 v22; // [rsp+8h] [rbp-38h]

  v3 = **a1;
  v4 = v3 + 48;
  v5 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 + 48 == v5 )
  {
    v7 = 0;
  }
  else
  {
    if ( !v5 )
      BUG();
    v6 = *(unsigned __int8 *)(v5 - 24);
    v7 = 0;
    v8 = v5 - 24;
    if ( (unsigned int)(v6 - 30) < 0xB )
      v7 = v8;
  }
  v9 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 48 == v9 )
  {
    v11 = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    v10 = *(unsigned __int8 *)(v9 - 24);
    v11 = 0;
    v12 = v9 - 24;
    if ( (unsigned int)(v10 - 30) < 0xB )
      v11 = v12;
  }
  v22 = a2 + 48;
  if ( !(unsigned __int8)sub_B46250(v11, v7, 0) )
    return 1;
  v13 = 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
  v14 = (_QWORD *)(v7 - v13);
  if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
    v14 = *(_QWORD **)(v7 - 8);
  v15 = 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v11 + 7) & 0x40) != 0 )
  {
    v16 = *(_QWORD **)(v11 - 8);
    v11 = (__int64)&v16[(unsigned __int64)v15 / 8];
  }
  else
  {
    v16 = (_QWORD *)(v11 - v15);
  }
  if ( v15 != v13 )
    return 1;
  while ( (_QWORD *)v11 != v16 )
  {
    if ( *v16 != *v14 )
      return 1;
    v16 += 4;
    v14 += 4;
  }
  v17 = *(_QWORD *)(v3 + 56);
  for ( i = 0; v17 != v4; ++i )
    v17 = *(_QWORD *)(v17 + 8);
  v19 = *(_QWORD *)(a2 + 56);
  for ( j = 0; v19 != v22; ++j )
    v19 = *(_QWORD *)(v19 + 8);
  return j != i;
}
