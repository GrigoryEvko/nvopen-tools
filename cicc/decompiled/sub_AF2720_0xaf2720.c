// Function: sub_AF2720
// Address: 0xaf2720
//
__int64 __fastcall sub_AF2720(__int64 a1)
{
  __int64 result; // rax

  result = sub_B757E0(a1, 1);
  *(_DWORD *)(a1 + 4) = result;
  return result;
}
