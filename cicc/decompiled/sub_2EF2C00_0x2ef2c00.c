// Function: sub_2EF2C00
// Address: 0x2ef2c00
//
__int64 __fastcall sub_2EF2C00(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rdx
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rsi

  v1 = *a1;
  v2 = *a1 >> 3;
  if ( (*(_BYTE *)a1 & 2) == 0 )
    return 8 * (v2 & 0xFFFFFFFFE0000000LL) + 1;
  v3 = HIWORD(v1);
  if ( (v1 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
    v3 = HIDWORD(v1);
  return 8 * ((v3 << 45) | v2 & 0x1FFFFFE00000LL) + 2;
}
