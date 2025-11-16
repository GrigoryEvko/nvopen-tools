// Function: sub_15F5F30
// Address: 0x15f5f30
//
__int64 __fastcall sub_15F5F30(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int16 v5; // dx
  _QWORD *v6; // rax
  __int64 v7; // r9
  __int64 *v8; // rcx
  _QWORD *v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rdi
  unsigned __int64 v12; // rsi
  __int64 v13; // rsi
  void *v14; // r14
  __int64 v15; // rax
  _BYTE *v16; // rsi
  __int64 v17; // rdx
  _BYTE *v18; // r13
  __int64 result; // rax

  v4 = *(_QWORD *)(a2 + 64);
  sub_15F1EA0(
    a1,
    *(_QWORD *)a2,
    54,
    a1 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
    *(_DWORD *)(a2 + 20) & 0xFFFFFFF,
    0);
  *(_QWORD *)(a1 + 64) = v4;
  v5 = *(_WORD *)(a1 + 18) & 0x8000;
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(a2 + 56);
  *(_WORD *)(a1 + 18) = v5 | *(_WORD *)(a2 + 18) & 0x7FFF;
  v6 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v7 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v8 = (__int64 *)(a2 - v7);
  if ( v7 )
  {
    v9 = &v6[(unsigned __int64)v7 / 8];
    do
    {
      v10 = *v8;
      if ( *v6 )
      {
        v11 = v6[1];
        v12 = v6[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v12 = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
      }
      *v6 = v10;
      if ( v10 )
      {
        v13 = *(_QWORD *)(v10 + 8);
        v6[1] = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = (unsigned __int64)(v6 + 1) | *(_QWORD *)(v13 + 16) & 3LL;
        v6[2] = (v10 + 8) | v6[2] & 3LL;
        *(_QWORD *)(v10 + 8) = v6;
      }
      v6 += 3;
      v8 += 3;
    }
    while ( v6 != v9 );
  }
  v14 = 0;
  if ( *(char *)(a1 + 23) < 0 )
    v14 = (void *)sub_1648A40(a1);
  if ( *(char *)(a2 + 23) < 0 )
  {
    v15 = sub_1648A40(a2);
    v16 = 0;
    v18 = (_BYTE *)(v15 + v17);
    if ( *(char *)(a2 + 23) < 0 )
      v16 = (_BYTE *)sub_1648A40(a2);
    if ( v16 != v18 )
      memmove(v14, v16, v18 - v16);
  }
  result = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1;
  return result;
}
