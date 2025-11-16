// Function: sub_15CA8D0
// Address: 0x15ca8d0
//
__int64 __fastcall sub_15CA8D0(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 96);
  *(_DWORD *)(a1 + 460) = result;
  return result;
}
