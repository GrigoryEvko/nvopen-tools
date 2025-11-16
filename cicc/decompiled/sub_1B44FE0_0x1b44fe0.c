// Function: sub_1B44FE0
// Address: 0x1b44fe0
//
__int64 __fastcall sub_1B44FE0(__int64 a1)
{
  char v1; // al
  __int64 *v2; // rax
  __int64 v3; // r12

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 27 )
  {
LABEL_2:
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v2 = *(__int64 **)(a1 - 8);
    else
      v2 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v3 = *v2;
    if ( !*v2 )
      BUG();
    if ( *(_BYTE *)(v3 + 16) > 0x17u )
      goto LABEL_6;
    return sub_15F20C0((_QWORD *)a1);
  }
  if ( v1 != 26 )
  {
    if ( v1 != 28 )
      return sub_15F20C0((_QWORD *)a1);
    goto LABEL_2;
  }
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 3 )
  {
    v3 = *(_QWORD *)(a1 - 72);
    if ( *(_BYTE *)(v3 + 16) > 0x17u )
    {
LABEL_6:
      sub_15F20C0((_QWORD *)a1);
      return sub_1AEB370(v3, 0);
    }
  }
  return sub_15F20C0((_QWORD *)a1);
}
