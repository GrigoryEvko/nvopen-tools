// Function: sub_15F5180
// Address: 0x15f5180
//
__int64 __fastcall sub_15F5180(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  char v5; // di
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 *v9; // rcx
  char v10; // si
  _QWORD *v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rdi
  unsigned __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rsi
  size_t v18; // rdx
  __int64 result; // rax

  sub_15F1EA0(a1, *(_QWORD *)a2, 53, 0, *(_DWORD *)(a2 + 20) & 0xFFFFFFF, 0);
  v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  *(_DWORD *)(a1 + 56) = v4;
  sub_1648880(a1, v4, 1);
  v5 = *(_BYTE *)(a1 + 23) & 0x40;
  if ( v5 )
    v6 = *(_QWORD **)(a1 - 8);
  else
    v6 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v8 = 3 * v7;
  v9 = (__int64 *)(a2 - 24 * v7);
  v10 = *(_BYTE *)(a2 + 23) & 0x40;
  if ( v10 )
    v9 = *(__int64 **)(a2 - 8);
  if ( v8 * 8 )
  {
    v11 = &v6[v8];
    do
    {
      v12 = *v9;
      if ( *v6 )
      {
        v13 = v6[1];
        v14 = v6[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v14 = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
      }
      *v6 = v12;
      if ( v12 )
      {
        v15 = *(_QWORD *)(v12 + 8);
        v6[1] = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = (unsigned __int64)(v6 + 1) | *(_QWORD *)(v15 + 16) & 3LL;
        v6[2] = (v12 + 8) | v6[2] & 3LL;
        *(_QWORD *)(v12 + 8) = v6;
      }
      v6 += 3;
      v9 += 3;
    }
    while ( v6 != v11 );
    v5 = *(_BYTE *)(a1 + 23) & 0x40;
    v10 = *(_BYTE *)(a2 + 23) & 0x40;
    v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  }
  if ( v5 )
    v16 = *(_QWORD *)(a1 - 8);
  else
    v16 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( v10 )
    v17 = *(_QWORD *)(a2 - 8);
  else
    v17 = a2 - 24 * v7;
  v18 = 8 * v7;
  if ( v18 )
    memmove(
      (void *)(v16 + 24LL * *(unsigned int *)(a1 + 56) + 8),
      (const void *)(v17 + 24LL * *(unsigned int *)(a2 + 56) + 8),
      v18);
  result = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1;
  return result;
}
