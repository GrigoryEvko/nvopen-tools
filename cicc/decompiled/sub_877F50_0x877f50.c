// Function: sub_877F50
// Address: 0x877f50
//
__int64 __fastcall sub_877F50(__int64 a1, __int64 a2)
{
  int v2; // r12d
  __int64 result; // rax

  v2 = *(_DWORD *)(a2 + 40);
  *(_DWORD *)(a2 + 40) = -1;
  result = sub_877D80(a1, (__int64 *)a2);
  *(_DWORD *)(a2 + 40) = v2;
  return result;
}
