// Function: sub_2649AA0
// Address: 0x2649aa0
//
__int64 __fastcall sub_2649AA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 result; // rax

  ++*(_QWORD *)a1;
  v2 = *(_QWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  v3 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = v2;
  LODWORD(v2) = *(_DWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = v3;
  LODWORD(v3) = *(_DWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 16) = v2;
  LODWORD(v2) = *(_DWORD *)(a2 + 20);
  *(_DWORD *)(a2 + 16) = v3;
  LODWORD(v3) = *(_DWORD *)(a1 + 20);
  *(_DWORD *)(a1 + 20) = v2;
  LODWORD(v2) = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a2 + 20) = v3;
  result = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 24) = v2;
  *(_DWORD *)(a2 + 24) = result;
  return result;
}
