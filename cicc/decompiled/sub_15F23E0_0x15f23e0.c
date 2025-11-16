// Function: sub_15F23E0
// Address: 0x15f23e0
//
__int64 __fastcall sub_15F23E0(__int64 a1, int a2)
{
  __int64 result; // rax

  result = *(_BYTE *)(a1 + 17) & 1
         | (2
          * ((a2 << 6)
           | (32 * a2) & 0xFFFFFFBF
           | (16 * a2) & 0xFFFFFF9F
           | (8 * a2) & 0xFFFFFF8F
           | (4 * a2) & 0xFFFFFF87
           | (2 * a2) & 0xFFFFFF83
           | a2 & 0xFFFFFF81
           | (*(_BYTE *)(a1 + 17) >> 1) & 0x80));
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a1 + 17) & 1
                      | (2
                       * (((_BYTE)a2 << 6)
                        | (32 * a2) & 0xBF
                        | (16 * a2) & 0x9F
                        | (8 * a2) & 0x8F
                        | (4 * a2) & 0x87
                        | (2 * a2) & 0x83
                        | a2 & 0x81
                        | (*(_BYTE *)(a1 + 17) >> 1) & 0x80));
  return result;
}
