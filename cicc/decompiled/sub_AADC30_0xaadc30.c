// Function: sub_AADC30
// Address: 0xaadc30
//
__int64 __fastcall sub_AADC30(__int64 a1, __int64 a2, __int64 *a3)
{
  int v3; // eax
  int v4; // eax
  __int64 result; // rax

  v3 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a2 + 8) = 0;
  *(_DWORD *)(a1 + 8) = v3;
  *(_QWORD *)a1 = *(_QWORD *)a2;
  v4 = *((_DWORD *)a3 + 2);
  *((_DWORD *)a3 + 2) = 0;
  *(_DWORD *)(a1 + 24) = v4;
  result = *a3;
  *(_QWORD *)(a1 + 16) = *a3;
  return result;
}
