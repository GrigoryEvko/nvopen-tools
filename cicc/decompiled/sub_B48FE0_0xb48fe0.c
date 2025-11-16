// Function: sub_B48FE0
// Address: 0xb48fe0
//
__int64 __fastcall sub_B48FE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  *(_DWORD *)(a1 + 4) &= 0xF8000000;
  *(_DWORD *)(a1 + 72) = a2;
  sub_BD2A10(a1, a2, 0);
  result = sub_BD6B50(a1, a3);
  *(_WORD *)(a1 + 2) &= ~1u;
  return result;
}
