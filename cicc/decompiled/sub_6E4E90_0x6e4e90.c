// Function: sub_6E4E90
// Address: 0x6e4e90
//
__int64 __fastcall sub_6E4E90(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  char v3; // r12
  int v4; // eax
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 128);
  v3 = *(_BYTE *)(a1 + 18);
  sub_6E4BC0(a1, a2);
  v4 = *(unsigned __int8 *)(a1 + 18);
  *(_QWORD *)(a1 + 128) = v2;
  result = v3 & 1 | v4 & 0xFFFFFFFE;
  *(_BYTE *)(a1 + 18) = result;
  return result;
}
