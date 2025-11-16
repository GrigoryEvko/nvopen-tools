// Function: sub_1F02230
// Address: 0x1f02230
//
__int64 __fastcall sub_1F02230(__int64 a1, int a2, int a3)
{
  __int64 result; // rax

  *(_DWORD *)(*(_QWORD *)(a1 + 40) + 4LL * a2) = a3;
  result = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(result + 4LL * a3) = a2;
  return result;
}
