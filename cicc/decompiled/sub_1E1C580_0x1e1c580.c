// Function: sub_1E1C580
// Address: 0x1e1c580
//
__int64 __fastcall sub_1E1C580(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned __int8 v8; // si
  unsigned __int64 v9; // rax
  _QWORD *v10; // rax
  const __m128i *v11; // rbx
  const __m128i *v12; // r14
  const __m128i *v13; // rdx
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v5 = *(_QWORD *)(a3 + 16);
  *(_BYTE *)(a1 + 44) = 0;
  *(_QWORD *)(a1 + 16) = v5;
  *(_WORD *)(a1 + 46) = 0;
  LOBYTE(v5) = *(_BYTE *)(a3 + 49);
  *(_QWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 49) = v5;
  v6 = *(_QWORD *)(a3 + 56);
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = v6;
  v7 = *(_QWORD *)(a3 + 64);
  *(_QWORD *)(a1 + 64) = v7;
  if ( v7 )
    sub_1623A60(a1 + 64, v7, 2);
  v8 = 0;
  if ( *(_DWORD *)(a3 + 40) && *(_DWORD *)(a3 + 40) != 1 )
  {
    _BitScanReverse64(&v9, *(unsigned int *)(a3 + 40) - 1LL);
    v8 = 64 - (v9 ^ 0x3F);
  }
  *(_BYTE *)(a1 + 44) = v8;
  v10 = sub_1E1A7D0(a2 + 232, v8, (__int64 *)(a2 + 120));
  v11 = *(const __m128i **)(a3 + 32);
  *(_QWORD *)(a1 + 32) = v10;
  v12 = (const __m128i *)((char *)v11 + 40 * *(unsigned int *)(a3 + 40));
  while ( v12 != v11 )
  {
    v13 = v11;
    v11 = (const __m128i *)((char *)v11 + 40);
    sub_1E1A9C0(a1, a2, v13);
  }
  result = *(_WORD *)(a3 + 46) & 0xFFF3 | *(unsigned __int16 *)(a1 + 46) & 0xCu;
  *(_WORD *)(a1 + 46) = *(_WORD *)(a3 + 46) & 0xFFF3 | *(_WORD *)(a1 + 46) & 0xC;
  return result;
}
