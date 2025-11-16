// Function: sub_1698630
// Address: 0x1698630
//
__int64 __fastcall sub_1698630(__int64 a1, __int64 a2)
{
  char v2; // al
  char v3; // al
  __int64 result; // rax

  v2 = *(_BYTE *)(a2 + 18) & 8 | *(_BYTE *)(a1 + 18) & 0xF7;
  *(_BYTE *)(a1 + 18) = v2;
  v3 = *(_BYTE *)(a2 + 18) & 7 | v2 & 0xF8;
  *(_BYTE *)(a1 + 18) = v3;
  result = v3 & 7;
  *(_WORD *)(a1 + 16) = *(_WORD *)(a2 + 16);
  if ( (_BYTE)result == 1 || (_BYTE)result && (_BYTE)result != 3 )
    return sub_16985E0(a1, a2);
  return result;
}
