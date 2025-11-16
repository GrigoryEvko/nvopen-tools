// Function: sub_688F10
// Address: 0x688f10
//
__int64 __fastcall sub_688F10(__int64 a1, char a2)
{
  char v2; // bl
  __int64 v3; // rax
  char v4; // si
  __int64 v5; // r12
  __int64 v6; // rax

  v2 = (a2 << 7) | 0x50;
  v3 = sub_6EAFA0(0);
  v4 = *(_BYTE *)(v3 + 49);
  *(_QWORD *)(v3 + 16) = a1;
  v5 = v3;
  *(_BYTE *)(v3 + 49) = v4 & 0x2F | v2;
  sub_6E1D20(a1);
  v6 = unk_4D03C50;
  if ( (*(_BYTE *)(unk_4D03C50 + 17LL) & 2) == 0 )
    return v5;
  *(_BYTE *)(a1 + 193) |= 0x40u;
  if ( (*(_BYTE *)(v6 + 17) & 1) == 0 )
    return v5;
  sub_7340D0(v5, 0, 0);
  return v5;
}
