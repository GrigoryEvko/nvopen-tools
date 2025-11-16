// Function: sub_E81B30
// Address: 0xe81b30
//
__int64 __fastcall sub_E81B30(__int64 a1, char a2, char a3)
{
  int v3; // eax
  __int64 result; // rax

  v3 = *(unsigned __int8 *)(a1 + 29);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  result = a3 & 0xF | v3 & 0xFFFFFFF0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = a2;
  *(_BYTE *)(a1 + 29) = result;
  return result;
}
