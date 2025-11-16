// Function: sub_691700
// Address: 0x691700
//
__int64 __fastcall sub_691700(__int64 a1, __int64 a2, char a3)
{
  char v3; // bl
  __int64 result; // rax

  v3 = 2 * (a3 & 1);
  result = sub_73DC30(6, a2, a1);
  *(_BYTE *)(result + 27) = *(_BYTE *)(result + 27) & 0xFD | v3;
  return result;
}
