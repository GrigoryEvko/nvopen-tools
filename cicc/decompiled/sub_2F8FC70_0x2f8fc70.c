// Function: sub_2F8FC70
// Address: 0x2f8fc70
//
__int64 __fastcall sub_2F8FC70(__int64 a1, int a2, int a3)
{
  __int64 result; // rax

  *(_DWORD *)(*(_QWORD *)(a1 + 320) + 4LL * a2) = a3;
  result = *(_QWORD *)(a1 + 296);
  *(_DWORD *)(result + 4LL * a3) = a2;
  return result;
}
