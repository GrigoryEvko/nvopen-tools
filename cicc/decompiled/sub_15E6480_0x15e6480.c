// Function: sub_15E6480
// Address: 0x15e6480
//
__int64 __fastcall sub_15E6480(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_15E6420(a1, a2);
  *(_BYTE *)(a1 + 33) = *(_BYTE *)(a2 + 33) & 0x1C | *(_BYTE *)(a1 + 33) & 0xE3;
  *(_BYTE *)(a1 + 80) = *(_BYTE *)(a2 + 80) & 2 | *(_BYTE *)(a1 + 80) & 0xFD;
  result = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a1 + 72) = result;
  return result;
}
