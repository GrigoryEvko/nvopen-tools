// Function: sub_EA12A0
// Address: 0xea12a0
//
__int64 __fastcall sub_EA12A0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 result; // rax

  v2 = *(unsigned __int8 *)(a1 + 9);
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)a1 = 0;
  result = v2 & 0xFFFFFF8F | 0x20;
  *(_BYTE *)(a1 + 9) = result;
  return result;
}
