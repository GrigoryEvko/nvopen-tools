// Function: sub_B15A30
// Address: 0xb15a30
//
__int64 __fastcall sub_B15A30(__int64 a1, __int64 *a2, _DWORD *a3, _DWORD *a4)
{
  __int64 v6; // rdx
  __int64 result; // rax

  *a2 = sub_B159E0((__int64 *)(a1 + 24), (__int64)a2);
  a2[1] = v6;
  *a3 = *(_DWORD *)(a1 + 32);
  result = *(unsigned int *)(a1 + 36);
  *a4 = result;
  return result;
}
