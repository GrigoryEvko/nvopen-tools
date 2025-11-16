// Function: sub_1BB9460
// Address: 0x1bb9460
//
__int64 __fastcall sub_1BB9460(__int64 a1, __int64 a2)
{
  int v2; // eax
  int v3; // edx
  __int64 result; // rax

  v2 = *(_DWORD *)(a2 + 88);
  *(_BYTE *)(a2 + 100) = 0;
  v3 = v2 - *(_DWORD *)(a2 + 92);
  *(_DWORD *)(a2 + 92) = v2;
  result = *(_QWORD *)(a2 + 8);
  *(_DWORD *)(result + 96) += v3;
  return result;
}
