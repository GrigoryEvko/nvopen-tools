// Function: sub_2246B50
// Address: 0x2246b50
//
__int64 __fastcall sub_2246B50(__int64 a1, __int64 a2, char a3, __int64 a4, unsigned int a5, unsigned __int64 a6)
{
  int v7; // r12d
  __int64 result; // rax

  v7 = *(_DWORD *)(a4 + 24);
  *(_DWORD *)(a4 + 24) = v7 & 0xFFFFBDB5 | 0x208;
  result = sub_2246920(a1, a2, a3, a4, a5, a6);
  *(_DWORD *)(a4 + 24) = v7;
  return result;
}
