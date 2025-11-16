// Function: sub_15F9C80
// Address: 0x15f9c80
//
__int64 __fastcall sub_15F9C80(__int64 a1, __int64 a2, __int16 a3, char a4, __int64 a5)
{
  __int16 v7; // r12
  __int64 v8; // rax
  __int64 result; // rax
  __int16 v10; // dx

  v7 = 2 * a3;
  v8 = sub_1643270(a2);
  result = sub_15F1EA0(a1, v8, 33, 0, 0, a5);
  v10 = *(_WORD *)(a1 + 18);
  *(_BYTE *)(a1 + 56) = a4;
  *(_WORD *)(a1 + 18) = v7 | v10 & 0x8001;
  return result;
}
