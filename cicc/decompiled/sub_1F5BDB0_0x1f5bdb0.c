// Function: sub_1F5BDB0
// Address: 0x1f5bdb0
//
__int64 __fastcall sub_1F5BDB0(__int64 a1, int a2, unsigned __int16 a3)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 264);
  *(_DWORD *)(result + 4LL * (a2 & 0x7FFFFFFF)) = a3;
  return result;
}
