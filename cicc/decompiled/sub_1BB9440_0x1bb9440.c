// Function: sub_1BB9440
// Address: 0x1bb9440
//
__int64 __fastcall sub_1BB9440(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdx
  __int64 result; // rax

  v2 = *(_DWORD *)(a2 + 92);
  v3 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a2 + 88) = -1;
  result = (unsigned int)~v2;
  *(_DWORD *)(v3 + 96) += result;
  *(_DWORD *)(a2 + 40) = 0;
  return result;
}
