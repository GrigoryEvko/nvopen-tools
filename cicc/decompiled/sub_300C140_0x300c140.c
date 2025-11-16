// Function: sub_300C140
// Address: 0x300c140
//
__int64 __fastcall sub_300C140(__int64 a1, int a2, int a3)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 56);
  *(_DWORD *)(result + 4LL * (a2 & 0x7FFFFFFF)) = a3;
  return result;
}
