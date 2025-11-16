// Function: sub_2626710
// Address: 0x2626710
//
__int64 __fastcall sub_2626710(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // rdi
  unsigned int v5; // ebx
  int v6; // r8d

  if ( (*(_BYTE *)(a1 + 32) & 0xF) == 1 )
    return 0;
  if ( sub_B2FC80(a1) )
    return 0;
  v3 = sub_BA91D0(*(_QWORD *)(a1 + 40), "CFI Canonical Jump Tables", 0x19u);
  if ( !v3 )
    return 1;
  v4 = *(_QWORD *)(v3 + 136);
  if ( !v4 )
    return 1;
  v5 = *(_DWORD *)(v4 + 32);
  if ( v5 <= 0x40 )
  {
    if ( !*(_QWORD *)(v4 + 24) )
      return sub_B2D620(a1, "cfi-canonical-jump-table", 0x18u);
    return 1;
  }
  v6 = sub_C444A0(v4 + 24);
  result = 1;
  if ( v5 == v6 )
    return sub_B2D620(a1, "cfi-canonical-jump-table", 0x18u);
  return result;
}
