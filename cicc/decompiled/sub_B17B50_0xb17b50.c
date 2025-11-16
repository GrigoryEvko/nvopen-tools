// Function: sub_B17B50
// Address: 0xb17b50
//
__int64 __fastcall sub_B17B50(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 88);
  *(_DWORD *)(a1 + 420) = result;
  return result;
}
