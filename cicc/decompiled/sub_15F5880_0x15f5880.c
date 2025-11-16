// Function: sub_15F5880
// Address: 0x15f5880
//
__int64 __fastcall sub_15F5880(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  *(_DWORD *)(a1 + 20) &= 0xF0000000;
  *(_DWORD *)(a1 + 56) = a2;
  sub_1648880(a1, a2, 0);
  result = sub_164B780(a1, a3);
  *(_WORD *)(a1 + 18) &= ~1u;
  return result;
}
