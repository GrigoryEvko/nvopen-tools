// Function: sub_160CD10
// Address: 0x160cd10
//
__int64 __fastcall sub_160CD10(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 160);
  *(_DWORD *)(a1 + 600) = result;
  return result;
}
