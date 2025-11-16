// Function: sub_38DBB60
// Address: 0x38dbb60
//
__int64 __fastcall sub_38DBB60(__int64 a1, int a2, int a3, __int16 a4, char a5, char a6, int a7)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(result + 1028) = a3;
  *(_DWORD *)(result + 1024) = a2;
  *(_WORD *)(result + 1032) = a4;
  *(_BYTE *)(result + 1034) = a5;
  *(_BYTE *)(result + 1035) = a6;
  *(_DWORD *)(result + 1036) = a7;
  *(_BYTE *)(result + 1040) = 1;
  return result;
}
