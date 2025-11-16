// Function: sub_15E4BE0
// Address: 0x15e4be0
//
__int64 __fastcall sub_15E4BE0(__int64 a1, __int64 a2)
{
  char v2; // al
  int v3; // eax
  unsigned int v4; // eax
  __int64 result; // rax

  v2 = *(_BYTE *)(a2 + 32) & 0x30 | *(_BYTE *)(a1 + 32) & 0xCF;
  *(_BYTE *)(a1 + 32) = v2;
  if ( (v2 & 0xFu) - 7 <= 1 || (v2 & 0x30) != 0 && (v2 & 0xF) != 9 )
  {
    v3 = *(unsigned __int8 *)(a1 + 33) | 0x40;
    *(_BYTE *)(a1 + 33) |= 0x40u;
  }
  else
  {
    v3 = *(unsigned __int8 *)(a1 + 33);
  }
  *(_BYTE *)(a1 + 32) = *(_BYTE *)(a2 + 32) & 0xC0 | *(_BYTE *)(a1 + 32) & 0x3F;
  v4 = *(_BYTE *)(a2 + 33) & 3 | v3 & 0xFFFFFFFC;
  *(_BYTE *)(a1 + 33) = v4;
  result = *(_BYTE *)(a2 + 33) & 0x40 | v4 & 0xFFFFFFBF;
  *(_BYTE *)(a1 + 33) = result;
  return result;
}
