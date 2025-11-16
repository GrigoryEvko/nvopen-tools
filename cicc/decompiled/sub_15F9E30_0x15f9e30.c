// Function: sub_15F9E30
// Address: 0x15f9e30
//
__int64 __fastcall sub_15F9E30(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  __int64 v5; // r9
  __int64 *v6; // rcx
  _QWORD *v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rdi
  unsigned __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 result; // rax

  sub_15F1EA0(
    a1,
    *(_QWORD *)a2,
    32,
    a1 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
    *(_DWORD *)(a2 + 20) & 0xFFFFFFF,
    0);
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a1 + 64) = *(_QWORD *)(a2 + 64);
  v4 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v5 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v6 = (__int64 *)(a2 - v5);
  if ( v5 )
  {
    v7 = &v4[(unsigned __int64)v5 / 8];
    do
    {
      v8 = *v6;
      if ( *v4 )
      {
        v9 = v4[1];
        v10 = v4[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v10 = v9;
        if ( v9 )
          *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
      }
      *v4 = v8;
      if ( v8 )
      {
        v11 = *(_QWORD *)(v8 + 8);
        v4[1] = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = (unsigned __int64)(v4 + 1) | *(_QWORD *)(v11 + 16) & 3LL;
        v4[2] = (v8 + 8) | v4[2] & 3LL;
        *(_QWORD *)(v8 + 8) = v4;
      }
      v4 += 3;
      v6 += 3;
    }
    while ( v4 != v7 );
  }
  result = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1;
  return result;
}
