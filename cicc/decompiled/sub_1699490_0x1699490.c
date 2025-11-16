// Function: sub_1699490
// Address: 0x1699490
//
__int64 __fastcall sub_1699490(__int64 a1)
{
  int v1; // edx
  __int64 result; // rax

  v1 = ~*(_BYTE *)(a1 + 18) & 8;
  result = v1 | *(_BYTE *)(a1 + 18) & 0xF7u;
  *(_BYTE *)(a1 + 18) = v1 | *(_BYTE *)(a1 + 18) & 0xF7;
  return result;
}
