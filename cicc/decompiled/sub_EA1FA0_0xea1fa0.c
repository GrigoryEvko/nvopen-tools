// Function: sub_EA1FA0
// Address: 0xea1fa0
//
__int64 __fastcall sub_EA1FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  sub_E8DA50(a1, a2, a3, a4, a5);
  result = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
  if ( (*(_BYTE *)(result + 176) & 2) != 0 )
    *(_WORD *)(a2 + 12) |= 0x100u;
  return result;
}
