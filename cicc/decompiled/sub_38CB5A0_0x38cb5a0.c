// Function: sub_38CB5A0
// Address: 0x38cb5a0
//
__int64 __fastcall sub_38CB5A0(__int64 a1, __int64 a2, __int16 a3, __int64 a4, __int64 a5)
{
  char v5; // al
  int v6; // edx
  char v7; // al
  __int64 result; // rax

  *(_DWORD *)a1 = 2;
  *(_QWORD *)(a1 + 8) = a5;
  v5 = *(_BYTE *)(a4 + 18);
  *(_WORD *)(a1 + 16) = a3;
  v6 = *(unsigned __int8 *)(a1 + 18);
  v7 = *(_BYTE *)(a4 + 359) | (2 * v5);
  *(_QWORD *)(a1 + 24) = a2;
  result = v6 & 0xFFFFFFFC | v7 & 3;
  *(_BYTE *)(a1 + 18) = result;
  return result;
}
