// Function: sub_223FE00
// Address: 0x223fe00
//
__int64 __fastcall sub_223FE00(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rax

  if ( !a2 || a3 < 0 )
    return a1;
  v4 = *(_BYTE **)(a1 + 72);
  *(_QWORD *)(a1 + 80) = 0;
  *v4 = 0;
  sub_223FD50(a1, a2, a3, 0);
  return a1;
}
