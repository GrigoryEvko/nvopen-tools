// Function: sub_164EE90
// Address: 0x164ee90
//
__int64 __fastcall sub_164EE90(__int64 a1)
{
  char v1; // al
  __int64 v2; // r8
  char v4; // dl
  __int64 v5; // rdi

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 29 )
  {
    v2 = *(_QWORD *)(a1 - 24);
    return sub_157ED20(v2);
  }
  v2 = 0;
  v4 = *(_BYTE *)(a1 + 18) & 1;
  if ( v1 == 34 )
  {
    if ( !v4 )
      return sub_157ED20(v2);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v5 = *(_QWORD *)(a1 - 8);
    else
      v5 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    return sub_157ED20(*(_QWORD *)(v5 + 24));
  }
  else
  {
    if ( !v4 )
      return sub_157ED20(v2);
    return sub_157ED20(*(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF))));
  }
}
