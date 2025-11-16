// Function: sub_31A2750
// Address: 0x31a2750
//
void __fastcall sub_31A2750(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // ebx
  bool v4; // bl
  unsigned __int16 v5; // ax
  char v6; // r8

  v2 = *(_QWORD *)(a1 + 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  v3 = *(_DWORD *)(v2 + 32);
  if ( v3 <= 0x40 )
    v4 = *(_QWORD *)(v2 + 24) == 0;
  else
    v4 = v3 == (unsigned int)sub_C444A0(v2 + 24);
  v5 = sub_A74840((_QWORD *)(a1 + 72), 0);
  v6 = 0;
  if ( HIBYTE(v5) )
    v6 = v5;
  sub_31A1E80(
    a1,
    *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)),
    *(_QWORD *)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))),
    *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))),
    v6,
    !v4);
}
