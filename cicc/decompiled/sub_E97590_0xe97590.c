// Function: sub_E97590
// Address: 0xe97590
//
__int64 __fastcall sub_E97590(__int64 a1, int a2, int a3, __int16 a4, char a5, char a6, int a7)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(result + 1780) = a3;
  *(_DWORD *)(result + 1776) = a2;
  *(_WORD *)(result + 1784) = a4;
  *(_BYTE *)(result + 1786) = a5;
  *(_BYTE *)(result + 1787) = a6;
  *(_DWORD *)(result + 1788) = a7;
  *(_BYTE *)(result + 1792) = 1;
  return result;
}
