// Function: sub_15F6900
// Address: 0x15f6900
//
__int64 __fastcall sub_15F6900(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int16 v6; // dx
  _QWORD *v7; // rax
  __int64 v8; // r9
  __int64 *v9; // rcx
  _QWORD *v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned __int64 v13; // rsi
  __int64 v14; // rsi
  void *v15; // r14
  __int64 v16; // rax
  _BYTE *v17; // rsi
  __int64 v18; // rdx
  _BYTE *v19; // r13
  __int64 result; // rax

  v4 = *(_QWORD *)(a2 + 64);
  sub_15F1EA0(a1, *(_QWORD *)a2, 5, a1 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF), *(_DWORD *)(a2 + 20) & 0xFFFFFFF, 0);
  v5 = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a1 + 64) = v4;
  v6 = *(_WORD *)(a1 + 18);
  *(_QWORD *)(a1 + 56) = v5;
  *(_WORD *)(a1 + 18) = v6 & 0x8000 | v6 & 3 | (4 * ((*(_WORD *)(a2 + 18) >> 2) & 0xDFFF));
  v7 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v8 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v9 = (__int64 *)(a2 - v8);
  if ( v8 )
  {
    v10 = &v7[(unsigned __int64)v8 / 8];
    do
    {
      v11 = *v9;
      if ( *v7 )
      {
        v12 = v7[1];
        v13 = v7[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v13 = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
      }
      *v7 = v11;
      if ( v11 )
      {
        v14 = *(_QWORD *)(v11 + 8);
        v7[1] = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = (unsigned __int64)(v7 + 1) | *(_QWORD *)(v14 + 16) & 3LL;
        v7[2] = (v11 + 8) | v7[2] & 3LL;
        *(_QWORD *)(v11 + 8) = v7;
      }
      v7 += 3;
      v9 += 3;
    }
    while ( v7 != v10 );
  }
  v15 = 0;
  if ( *(char *)(a1 + 23) < 0 )
    v15 = (void *)sub_1648A40(a1);
  if ( *(char *)(a2 + 23) < 0 )
  {
    v16 = sub_1648A40(a2);
    v17 = 0;
    v19 = (_BYTE *)(v16 + v18);
    if ( *(char *)(a2 + 23) < 0 )
      v17 = (_BYTE *)sub_1648A40(a2);
    if ( v17 != v19 )
      memmove(v15, v17, v19 - v17);
  }
  result = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1;
  return result;
}
