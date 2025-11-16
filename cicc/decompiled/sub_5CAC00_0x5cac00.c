// Function: sub_5CAC00
// Address: 0x5cac00
//
__int64 __fastcall sub_5CAC00(__int64 a1, __int64 a2, char a3)
{
  char v4; // al
  __int64 v5; // rax

  if ( a3 != 11 )
    return a2;
  if ( (*(_QWORD *)(a2 + 200) & 0x8000001000000LL) == 0x8000000000000LL )
  {
    v5 = sub_8258E0(a2, 0);
    sub_6865F0(3469, a1 + 56, "__global__", v5);
  }
  else
  {
    sub_5C8600(a1, a2);
  }
  v4 = *(_BYTE *)(a2 + 198);
  if ( (v4 & 0x20) == 0 )
    return a2;
  *(_BYTE *)(a2 + 198) = v4 | 0x40;
  return a2;
}
