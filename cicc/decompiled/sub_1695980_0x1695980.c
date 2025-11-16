// Function: sub_1695980
// Address: 0x1695980
//
__int64 __fastcall sub_1695980(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v4; // rdx
  unsigned __int64 v5; // rax

  v1 = sub_16321C0(a1, (__int64)"__llvm_profile_raw_version", 26, 1);
  if ( !v1 )
    return 0;
  v2 = v1;
  if ( sub_15E4F60(v1) )
    return 0;
  if ( (*(_BYTE *)(v2 + 32) & 0xFu) - 7 <= 1 )
    return 0;
  if ( sub_15E4F60(v2) )
    return 0;
  v4 = *(_QWORD *)(v2 - 24);
  if ( !v4 )
    return 0;
  if ( *(_BYTE *)(v4 + 16) != 13 )
    BUG();
  v5 = *(_QWORD *)(v4 + 24);
  if ( *(_DWORD *)(v4 + 32) > 0x40u )
    v5 = *(_QWORD *)v5;
  return HIBYTE(v5) & 1;
}
