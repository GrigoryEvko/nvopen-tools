// Function: sub_B4B340
// Address: 0xb4b340
//
__int64 __fastcall sub_B4B340(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 *v8; // rcx
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rsi
  void *v15; // r14
  __int64 v16; // rax
  _BYTE *v17; // rsi
  __int64 v18; // rdx
  _BYTE *v19; // r13
  __int64 result; // rax

  v5 = *(_QWORD *)(a2 + 80);
  sub_B44260(a1, *(_QWORD *)(a2 + 8), 11, a3, 0, 0);
  v6 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a1 + 80) = v5;
  *(_QWORD *)(a1 + 72) = v6;
  v7 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v8 = (__int64 *)(a2 - v7);
  v9 = *(_DWORD *)(a1 + 4);
  *(_WORD *)(a1 + 2) = *(_WORD *)(a2 + 2) & 0xFFC | *(_WORD *)(a1 + 2) & 0xF003;
  v10 = a1 - 32LL * (v9 & 0x7FFFFFF);
  if ( v7 )
  {
    v11 = v10 + v7;
    do
    {
      v12 = *v8;
      if ( *(_QWORD *)v10 )
      {
        v13 = *(_QWORD *)(v10 + 8);
        **(_QWORD **)(v10 + 16) = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = *(_QWORD *)(v10 + 16);
      }
      *(_QWORD *)v10 = v12;
      if ( v12 )
      {
        v14 = *(_QWORD *)(v12 + 16);
        *(_QWORD *)(v10 + 8) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = v10 + 8;
        *(_QWORD *)(v10 + 16) = v12 + 16;
        *(_QWORD *)(v12 + 16) = v10;
      }
      v10 += 32;
      v8 += 4;
    }
    while ( v10 != v11 );
  }
  v15 = 0;
  if ( *(char *)(a1 + 7) < 0 )
    v15 = (void *)sub_BD2BC0(a1);
  if ( *(char *)(a2 + 7) < 0 )
  {
    v16 = sub_BD2BC0(a2);
    v17 = 0;
    v19 = (_BYTE *)(v16 + v18);
    if ( *(char *)(a2 + 7) < 0 )
      v17 = (_BYTE *)sub_BD2BC0(a2);
    if ( v17 != v19 )
      memmove(v15, v17, v19 - v17);
  }
  *(_BYTE *)(a1 + 1) = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1;
  result = *(unsigned int *)(a2 + 88);
  *(_DWORD *)(a1 + 88) = result;
  return result;
}
