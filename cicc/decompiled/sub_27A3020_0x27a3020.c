// Function: sub_27A3020
// Address: 0x27a3020
//
__int64 __fastcall sub_27A3020(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  unsigned __int64 v4; // rcx
  int v5; // r8d
  int v6; // eax
  unsigned __int64 v7; // rcx
  int v8; // r8d
  int v9; // eax
  int v10; // ecx
  __int64 v11; // rdi
  unsigned __int64 v12; // rsi
  unsigned __int8 v13; // al

  result = *a3;
  switch ( (_BYTE)result )
  {
    case '=':
      _BitScanReverse64(&v4, 1LL << (*(_WORD *)(a2 + 2) >> 1));
      v5 = 63 - (v4 ^ 0x3F);
      _BitScanReverse64(&v4, 1LL << (*((_WORD *)a3 + 1) >> 1));
      LODWORD(v4) = v4 ^ 0x3F;
      v6 = 63 - v4;
      if ( (unsigned __int8)(63 - v4) > (unsigned __int8)v5 )
        v6 = v5;
      result = *((_WORD *)a3 + 1) & 0xFF81 | (unsigned int)(2 * v6);
      *((_WORD *)a3 + 1) = result;
      break;
    case '>':
      _BitScanReverse64(&v7, 1LL << (*(_WORD *)(a2 + 2) >> 1));
      v8 = 63 - (v7 ^ 0x3F);
      _BitScanReverse64(&v7, 1LL << (*((_WORD *)a3 + 1) >> 1));
      LODWORD(v7) = v7 ^ 0x3F;
      v9 = 63 - v7;
      if ( (unsigned __int8)(63 - v7) > (unsigned __int8)v8 )
        v9 = v8;
      result = *((_WORD *)a3 + 1) & 0xFF81 | (unsigned int)(2 * v9);
      *((_WORD *)a3 + 1) = result;
      break;
    case '<':
      v10 = *((unsigned __int16 *)a3 + 1);
      _BitScanReverse64((unsigned __int64 *)&v11, 1LL << *(_WORD *)(a2 + 2));
      _BitScanReverse64(&v12, 1LL << v10);
      v13 = 63 - (v12 ^ 0x3F);
      if ( v13 <= (unsigned __int8)(63 - (v11 ^ 0x3F)) )
        v13 = 63 - (v11 ^ 0x3F);
      result = v10 & 0xFFFFFFC0 | v13;
      *((_WORD *)a3 + 1) = result;
      break;
  }
  return result;
}
