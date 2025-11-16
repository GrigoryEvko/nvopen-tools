// Function: sub_7E2300
// Address: 0x7e2300
//
__int64 __fastcall sub_7E2300(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  sub_7264E0(a1, 1);
  result = sub_73D8E0(a1, 5u, a3, 0, a2);
  *(_BYTE *)(a1 + 27) |= 2u;
  return result;
}
