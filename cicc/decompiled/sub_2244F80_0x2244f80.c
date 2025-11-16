// Function: sub_2244F80
// Address: 0x2244f80
//
__int64 __fastcall sub_2244F80(__int64 a1, char *a2, __int64 a3, int a4, __int64 a5, _DWORD *a6, __int64 a7, int *a8)
{
  __int64 result; // rax

  result = sub_2244D30(a6, a4, a2, a3, a7, a7 + 4LL * *a8) - a6;
  *a8 = result;
  return result;
}
