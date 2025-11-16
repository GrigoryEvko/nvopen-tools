// Function: sub_1869D60
// Address: 0x1869d60
//
__int64 __fastcall sub_1869D60(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 160);
  *(_DWORD *)(a1 + 600) = result;
  return result;
}
