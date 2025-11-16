// Function: sub_39F1B60
// Address: 0x39f1b60
//
__int64 __fastcall sub_39F1B60(__int64 a1, int a2, int a3, int a4, int a5)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 264);
  *(_BYTE *)(result + 2056) = 1;
  *(_DWORD *)(result + 2060) = a2;
  *(_DWORD *)(result + 2064) = a3;
  *(_DWORD *)(result + 2068) = a4;
  *(_DWORD *)(result + 2072) = a5;
  return result;
}
