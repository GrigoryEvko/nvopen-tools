// Function: sub_1698360
// Address: 0x1698360
//
__int64 __fastcall sub_1698360(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_1698320((_QWORD *)a1, a2);
  result = *(_BYTE *)(a1 + 18) & 0xF0 | 3u;
  *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF0 | 3;
  return result;
}
