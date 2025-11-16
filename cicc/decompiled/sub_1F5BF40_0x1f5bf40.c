// Function: sub_1F5BF40
// Address: 0x1f5bf40
//
__int64 __fastcall sub_1F5BF40(__int64 a1, int a2, int a3)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 288);
  *(_DWORD *)(result + 4LL * (a2 & 0x7FFFFFFF)) = a3;
  return result;
}
