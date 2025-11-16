// Function: sub_B4A470
// Address: 0xb4a470
//
__int64 __fastcall sub_B4A470(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // r13
  __int64 v6; // rax
  __int16 v7; // dx
  __int64 v8; // rdi
  int v9; // edx
  __int64 *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rsi
  void *v16; // r14
  __int64 v17; // rax
  _BYTE *v18; // rsi
  __int64 v19; // rdx
  _BYTE *v20; // r13
  __int64 result; // rax

  v5 = *(_QWORD *)(a2 + 80);
  sub_B44260(a1, *(_QWORD *)(a2 + 8), 56, a3, 0, 0);
  v6 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a1 + 80) = v5;
  v7 = *(_WORD *)(a1 + 2);
  *(_QWORD *)(a1 + 72) = v6;
  v8 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  LOWORD(v6) = *(_WORD *)(a2 + 2) & 3 | v7 & 0xF000 | *(_WORD *)(a2 + 2) & 0xFFC;
  v9 = *(_DWORD *)(a1 + 4);
  v10 = (__int64 *)(a2 - v8);
  *(_WORD *)(a1 + 2) = v6;
  v11 = a1 - 32LL * (v9 & 0x7FFFFFF);
  if ( v8 )
  {
    v12 = v11 + v8;
    do
    {
      v13 = *v10;
      if ( *(_QWORD *)v11 )
      {
        v14 = *(_QWORD *)(v11 + 8);
        **(_QWORD **)(v11 + 16) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = *(_QWORD *)(v11 + 16);
      }
      *(_QWORD *)v11 = v13;
      if ( v13 )
      {
        v15 = *(_QWORD *)(v13 + 16);
        *(_QWORD *)(v11 + 8) = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = v11 + 8;
        *(_QWORD *)(v11 + 16) = v13 + 16;
        *(_QWORD *)(v13 + 16) = v11;
      }
      v11 += 32;
      v10 += 4;
    }
    while ( v11 != v12 );
  }
  v16 = 0;
  if ( *(char *)(a1 + 7) < 0 )
    v16 = (void *)sub_BD2BC0(a1);
  if ( *(char *)(a2 + 7) < 0 )
  {
    v17 = sub_BD2BC0(a2);
    v18 = 0;
    v20 = (_BYTE *)(v17 + v19);
    if ( *(char *)(a2 + 7) < 0 )
      v18 = (_BYTE *)sub_BD2BC0(a2);
    if ( v18 != v20 )
      memmove(v16, v18, v20 - v18);
  }
  result = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1u;
  *(_BYTE *)(a1 + 1) = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1;
  return result;
}
