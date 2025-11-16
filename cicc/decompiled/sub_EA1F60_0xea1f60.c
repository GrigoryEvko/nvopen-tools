// Function: sub_EA1F60
// Address: 0xea1f60
//
__int64 __fastcall sub_EA1F60(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  sub_E8DC70(a1, a2, a3);
  result = *(_QWORD *)(a1[36] + 8LL);
  if ( (*(_BYTE *)(result + 176) & 2) != 0 )
    *(_WORD *)(a2 + 12) |= 0x100u;
  return result;
}
