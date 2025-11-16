// Function: sub_160CD20
// Address: 0x160cd20
//
__int64 __fastcall sub_160CD20(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 600);
  *(_DWORD *)(a1 + 160) = result;
  return result;
}
