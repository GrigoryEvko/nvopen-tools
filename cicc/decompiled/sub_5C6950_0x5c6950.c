// Function: sub_5C6950
// Address: 0x5c6950
//
__int64 __fastcall sub_5C6950(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // rax

  v2 = 0;
  if ( !unk_4D046C4 )
  {
    sub_684AA0(7, 3659, a1 + 56);
    v2 = 1;
  }
  if ( unk_4D045E8 <= 0x4Fu )
  {
    sub_684AA0(7, 3660, a1 + 56);
    v2 = 1;
  }
  if ( !a2 )
    return a2;
  v3 = *(_QWORD *)(a2 + 152);
  if ( !v3 )
    goto LABEL_7;
  while ( *(_BYTE *)(v3 + 140) == 12 )
    v3 = *(_QWORD *)(v3 + 160);
  if ( (*(_BYTE *)(*(_QWORD *)(v3 + 168) + 16LL) & 1) == 0 )
  {
LABEL_7:
    if ( v2 != 1 )
      *(_BYTE *)(a2 + 199) |= 4u;
    return a2;
  }
  sub_684AA0(7, 3662, a1 + 56);
  return a2;
}
