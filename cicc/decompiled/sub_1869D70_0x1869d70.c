// Function: sub_1869D70
// Address: 0x1869d70
//
__int64 __fastcall sub_1869D70(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 600);
  *(_DWORD *)(a1 + 160) = result;
  return result;
}
