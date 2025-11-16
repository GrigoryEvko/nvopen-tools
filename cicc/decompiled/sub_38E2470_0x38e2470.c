// Function: sub_38E2470
// Address: 0x38e2470
//
__int64 __fastcall sub_38E2470(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 result; // rax

  v2 = *(unsigned __int8 *)(a1 + 9);
  *(_QWORD *)a1 &= 7uLL;
  *(_QWORD *)(a1 + 24) = a2;
  result = v2 & 0xFFFFFFF3 | 8;
  *(_BYTE *)(a1 + 9) = result;
  return result;
}
